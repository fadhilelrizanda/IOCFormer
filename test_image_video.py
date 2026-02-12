from __future__ import division

import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from PIL import Image

import util.misc as utils
from config import return_args, args
from Networks.CDETR import build_model


def pad_to_multiple_pil(img_pil: Image.Image, multiple: int):
    """Pad bottom/right so H,W are multiples of `multiple`."""
    w, h = img_pil.size
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % multiple)) % multiple

    if pad_h == 0 and pad_w == 0:
        return img_pil, (0, 0)

    # Pad on right and bottom
    new_w = w + pad_w
    new_h = h + pad_h
    padded = Image.new('RGB', (new_w, new_h), (0, 0, 0))
    padded.paste(img_pil, (0, 0))
    
    return padded, (pad_h, pad_w)


def split_into_patches_pil(img_pil: Image.Image, crop_size: int, transform):
    """
    Convert PIL image to patches, matching the dataloader's logic exactly.
    """
    # Apply transform (ToTensor + Normalize) - same as dataloader
    img_t = transform(img_pil)  # [3, H, W]
    
    width, height = img_t.shape[2], img_t.shape[1]
    num_w = int(width / crop_size)
    num_h = int(height / crop_size)
    
    # Match dataset.py patch splitting logic exactly
    img_t = img_t.view(3, num_h, crop_size, width).view(3, num_h, crop_size, num_w, crop_size)
    img_t = img_t.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, crop_size, crop_size)
    patches = img_t.permute(1, 0, 2, 3).contiguous()  # [N, 3, crop, crop]
    
    return patches, num_h, num_w, height, width


def show_map(out_pointes, frame_bgr, width, height, crop_size, num_h, num_w, threshold=0.25):
    """
    Builds a stitched point map and draws points.
    out_pointes: (Npatch, 1, Q, 3) with [conf, x, y] in patch coords.
    """
    kpoint_list = []

    for i in range(len(out_pointes)):
        out_value = out_pointes[i].squeeze(0)[:, 0].data.cpu().numpy()
        out_point = out_pointes[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()

        k = np.zeros((crop_size, crop_size), dtype=np.float32)

        for j in range(len(out_point)):
            if out_value[j] < threshold:
                continue
            x = int(out_point[j][0])
            y = int(out_point[j][1])
            if 0 <= x < crop_size and 0 <= y < crop_size:
                k[x, y] = 1.0

        kpoint_list.append(k)

    kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
    kpoint = (
        kpoint.view(num_h, num_w, crop_size, crop_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(num_h, crop_size, width)
        .view(height, width)
        .cpu()
        .numpy()
    )

    density_map = gaussian_filter(kpoint.copy(), 6)
    if np.max(density_map) > 0:
        density_map = density_map / np.max(density_map) * 255
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)

    pred_coor = np.nonzero(kpoint)
    count = len(pred_coor[0])

    point_map = np.zeros((kpoint.shape[0], kpoint.shape[1], 3), dtype=np.uint8) + 255
    frame_drawn = frame_bgr.copy()

    for i in range(count):
        w = int(pred_coor[1][i])
        h = int(pred_coor[0][i])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)
        cv2.circle(frame_drawn, (w, h), 3, (0, 0, 255), -1)

    return point_map, density_map, frame_drawn, count


def build_and_load_model():
    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()

    gpu_ids = [int(x) for x in str(args.gpu_id).split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        gpu_ids = [0]
    model = nn.DataParallel(model, device_ids=gpu_ids)

    if args.pre:
        if os.path.isfile(args.pre):
            ckpt = torch.load(args.pre, map_location="cpu")
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            # Match demo_video.py: rename bbox->point
            new_state = OrderedDict()
            for k, v in state.items():
                name = k.replace("bbox", "point")
                new_state[name] = v

            print(f"=> loading checkpoint '{args.pre}'")
            model.load_state_dict(new_state, strict=False)
        else:
            raise FileNotFoundError(f"=> no checkpoint found at '{args.pre}'")
    else:
        raise ValueError("You must pass --pre /path/to/checkpoint.pth")

    model.eval()
    return model


@torch.no_grad()
def infer_and_save_video(model):
    if not hasattr(args, "video_path") or not args.video_path:
        raise ValueError("You must pass --video_path /path/to/video.mp4")

    out_dir = args.save_path
    os.makedirs(out_dir, exist_ok=True)

    video_path = args.video_path
    base = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(out_dir, f"{base}_output.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    crop_size = int(args.crop_size)
    threshold = float(args.threshold)
    num_queries = int(args.num_queries)

    # Create transform matching the dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    frame_idx = 0
    total_count = 0

    while True:
        ret, img_bgr = cap.read()
        if not ret:
            break

        # Convert BGR to RGB and create PIL Image (matching dataloader)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Pad to multiple of crop_size
        img_pil_padded, (pad_h, pad_w) = pad_to_multiple_pil(img_pil, crop_size)
        
        # Split into patches
        patches, num_h, num_w, H, W = split_into_patches_pil(img_pil_padded, crop_size, transform)
        patches = patches.cuda()

        num_patches = patches.shape[0]

        outputs = model(patches)
        if isinstance(outputs, list):
            outputs = outputs[0]
        if not isinstance(outputs, dict):
            raise RuntimeError(f"Unexpected output type: {type(outputs)}")

        out_logits, out_point = outputs["pred_logits"], outputs["pred_points"]

        # ========== MATCH TEST.PY COUNTING LOGIC ==========
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits_reshaped = out_logits.view(1, -1, 2)
        
        topk_k = num_patches * num_queries
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits_reshaped.shape[0], -1),
            topk_k,
            dim=1
        )
        
        # Count predictions above threshold
        count = 0
        for k in range(topk_values.shape[0]):
            sub_count = topk_values[k, :]
            sub_count = sub_count.clone()
            sub_count[sub_count < threshold] = 0
            sub_count[sub_count > 0] = 1
            count += torch.sum(sub_count).item()
        
        # For visualization - map points back to patches
        topk_points_idx = topk_indexes // 2
        
        # Extract x,y coordinates only
        out_point_xy = out_point[:, :, :2]
        out_point_flat = out_point_xy.reshape(-1, 2)
        out_point_selected = out_point_flat[topk_points_idx[0]]
        out_point_selected = out_point_selected * crop_size
        
        # Reconstruct for visualization
        value_points = torch.zeros(num_patches, num_queries, 3).cuda()
        
        for i in range(topk_k):
            if topk_values[0, i] < threshold:
                continue
            
            flat_idx = topk_points_idx[0, i].item()
            patch_idx = flat_idx // num_queries
            query_idx = flat_idx % num_queries
            
            if patch_idx < num_patches and query_idx < num_queries:
                value_points[patch_idx, query_idx, 0] = topk_values[0, i]
                value_points[patch_idx, query_idx, 1:] = out_point_selected[i]
        
        value_points = value_points.unsqueeze(1)
        # ========== END MATCHING LOGIC ==========

        # Convert padded PIL image back to BGR for visualization
        img_padded_np = np.array(img_pil_padded)
        img_padded_bgr = cv2.cvtColor(img_padded_np, cv2.COLOR_RGB2BGR)

        point_map, density_map, drawn, count_visual = show_map(
            value_points, img_padded_bgr.copy(), W, H, crop_size, num_h, num_w, threshold=threshold
        )

        drawn_vis = drawn.copy()
        cv2.putText(drawn_vis, f"Count: {int(count)}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Crop back to original size if padded
        if pad_h > 0 or pad_w > 0:
            oh, ow = height, width
            drawn_vis = drawn_vis[:oh, :ow]

        out_video.write(drawn_vis)
        total_count += count
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames... (last count: {int(count)})")

    cap.release()
    out_video.release()
    print(f"Saved output video: {output_video_path}")
    print(f"Processed {frame_idx} frames. Total predicted count (sum over frames): {int(total_count)}")


def main():
    model = build_and_load_model()
    if hasattr(args, "video_path") and args.video_path:
        infer_and_save_video(model)
    else:
        raise ValueError("You must pass --video_path /path/to/video.mp4")


if __name__ == "__main__":
    main()