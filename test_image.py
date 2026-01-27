from __future__ import division

import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.ndimage import gaussian_filter  # fixed deprecation

import util.misc as utils
from config import return_args, args
from Networks.CDETR import build_model


# Keep consistent with demo_video.py
img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def pad_to_multiple(img_bgr: np.ndarray, multiple: int):
    """Pad bottom/right so H,W are multiples of `multiple`."""
    h, w = img_bgr.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    if pad_h == 0 and pad_w == 0:
        return img_bgr, (0, 0)

    padded = cv2.copyMakeBorder(
        img_bgr, 0, pad_h, 0, pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded, (pad_h, pad_w)


def split_into_patches(img_bgr: np.ndarray, crop_size: int):
    """
    Convert to tensor + normalize, then split into (N,3,crop,crop) patches.
    Matches demo_video.py’s reshape logic.
    """
    img_t = to_tensor(img_bgr)     # (3,H,W), float [0,1]
    img_t = img_transform(img_t)

    width, height = img_t.shape[2], img_t.shape[1]
    num_w = int(width / crop_size)
    num_h = int(height / crop_size)

    img_t = img_t.view(3, num_h, crop_size, width).view(3, num_h, crop_size, num_w, crop_size)
    img_t = img_t.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, crop_size, crop_size)
    patches = img_t.permute(1, 0, 2, 3).contiguous()  # (N,3,crop,crop)

    return patches, num_h, num_w, height, width


def show_map(out_pointes, frame_bgr, width, height, crop_size, num_h, num_w, threshold=0.25):
    """
    Adapted from demo_video.py. Builds a stitched point map and draws points.
    out_pointes: (Npatch, 1, Q, 3) with [conf, x, y] in patch coords.
    """
    kpoint_list = []

    for i in range(len(out_pointes)):
        out_value = out_pointes[i].squeeze(0)[:, 0].data.cpu().numpy()
        out_point = out_pointes[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()

        k = np.zeros((crop_size, crop_size), dtype=np.float32)

        for j in range(len(out_point)):
            if out_value[j] < threshold:
                break
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
        cv2.circle(frame_drawn, (w, h), 3, (0, 255, 50), -1)

    return point_map, density_map, frame_drawn, count


def build_and_load_model():
    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()

    # Use your gpu_id string style (e.g., "0,1"), but for single inference we typically use first GPU
    # DataParallel expects list of ints
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
def infer_and_save_single_image(model):
    if not args.image_path:
        raise ValueError("You must pass --image_path /path/to/image.jpg")

    # use save_path as output directory
    out_dir = args.save_path
    os.makedirs(out_dir, exist_ok=True)

    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {args.image_path}")

    base = os.path.splitext(os.path.basename(args.image_path))[0]

    crop_size = int(args.crop_size)
    threshold = float(args.threshold)
    num_queries = int(args.num_queries)  # from config.py (default 500)

    # pad to multiple of crop_size
    img_pad, (pad_h, pad_w) = pad_to_multiple(img_bgr, crop_size)

    patches, num_h, num_w, H, W = split_into_patches(img_pad, crop_size)
    patches = patches.cuda()

    outputs = model(patches)
    if isinstance(outputs, list):
        outputs = outputs[0]
    if not isinstance(outputs, dict):
        raise RuntimeError(f"Unexpected output type: {type(outputs)}")

    out_logits, out_point = outputs["pred_logits"], outputs["pred_points"]

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_queries, dim=1)

    topk_points = topk_indexes // out_logits.shape[2]
    out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
    out_point = out_point * crop_size

    value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)

    point_map, density_map, drawn, count = show_map(
        value_points, img_pad.copy(), W, H, crop_size, num_h, num_w, threshold=threshold
    )

    # annotate
    drawn_vis = drawn.copy()
    cv2.putText(drawn_vis, f"Count: {count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # crop back to original size if padded
    if pad_h > 0 or pad_w > 0:
        oh, ow = img_bgr.shape[:2]
        drawn_vis = drawn_vis[:oh, :ow]
        point_map = point_map[:oh, :ow]
        density_map = density_map[:oh, :ow]

    out_points_path = os.path.join(out_dir, f"{base}_pred_points.png")
    out_pointmap_path = os.path.join(out_dir, f"{base}_point_map.png")
    out_density_path = os.path.join(out_dir, f"{base}_density_map.png")

    cv2.imwrite(out_points_path, drawn_vis)
    cv2.imwrite(out_pointmap_path, point_map)
    cv2.imwrite(out_density_path, density_map)

    print("Saved:")
    print(" ", out_points_path)
    print(" ", out_pointmap_path)
    print(" ", out_density_path)
    print("Predicted count:", count)


def main():
    model = build_and_load_model()
    infer_and_save_single_image(model)


if __name__ == "__main__":
    main()
