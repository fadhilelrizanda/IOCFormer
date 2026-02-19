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
import torchvision.transforms.functional as TF
import math


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
        cv2.circle(frame_drawn, (w, h), 3, (0, 255, 50), -1)

    return point_map, density_map, frame_drawn, count


def grid_counts_from_density(density, rows, cols):
    if isinstance(density, torch.Tensor):
        dens = density.detach().cpu().numpy()
    else:
        dens = np.array(density)
    if dens.ndim == 3:
        dens = dens.squeeze(0)
    H, W = dens.shape
    ch = int(np.ceil(H / rows))
    cw = int(np.ceil(W / cols))
    counts = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        y0 = r * ch
        y1 = min((r + 1) * ch, H)
        for c in range(cols):
            x0 = c * cw
            x1 = min((c + 1) * cw, W)
            counts[r, c] = float(dens[y0:y1, x0:x1].sum())
    return counts


def mask_image_by_cells(img_np, dense_mask_cells, rows, cols, density_shape, mode='blur'):
    out = img_np.copy()
    den_h, den_w = density_shape
    img_h, img_w = img_np.shape[:2]
    ch = int(np.ceil(den_h / rows))
    cw = int(np.ceil(den_w / cols))
    for r in range(rows):
        for c in range(cols):
            if not dense_mask_cells[r, c]:
                continue
            y0 = r * ch
            y1 = min((r + 1) * ch, den_h)
            x0 = c * cw
            x1 = min((c + 1) * cw, den_w)
            y0_img = int(y0 * img_h / den_h)
            y1_img = int(y1 * img_h / den_h)
            x0_img = int(x0 * img_w / den_w)
            x1_img = int(x1 * img_w / den_w)
            if y1_img <= y0_img or x1_img <= x0_img:
                continue
            patch = out[y0_img:y1_img, x0_img:x1_img]
            if patch.size == 0:
                continue
            if mode == 'blur':
                kh = max(3, ((y1_img - y0_img) // 10) | 1)
                kw = max(3, ((x1_img - x0_img) // 10) | 1)
                k_h = kh if kh % 2 == 1 else kh + 1
                k_w = kw if kw % 2 == 1 else kw + 1
                try:
                    blurred = cv2.GaussianBlur(patch, (k_w, k_h), 0)
                except:
                    blurred = cv2.blur(patch, (5,5))
                out[y0_img:y1_img, x0_img:x1_img] = blurred
            elif mode == 'black':
                out[y0_img:y1_img, x0_img:x1_img] = (0, 0, 0)
            else:
                mean_color = [int(x) for x in cv2.mean(patch)[:3]]
                out[y0_img:y1_img, x0_img:x1_img] = mean_color
    return out


def draw_grid_lines(img, rows, cols, color=(0, 255, 0), thickness=1):
    h, w = img.shape[:2]
    for r in range(1, rows):
        y = int(r * h / rows)
        cv2.line(img, (0, y), (w, y), color, thickness)
    for c in range(1, cols):
        x = int(c * w / cols)
        cv2.line(img, (x, 0), (x, h), color, thickness)
    return img


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
def infer_and_save_single_image(model):
    if not args.image_path:
        raise ValueError("You must pass --image_path /path/to/image.jpg")

    out_dir = args.save_path
    os.makedirs(out_dir, exist_ok=True)

    # Load image using PIL (same as dataloader and H5 generation)
    img_pil = Image.open(args.image_path).convert('RGB')
    
    base = os.path.splitext(os.path.basename(args.image_path))[0]

    crop_size = int(args.crop_size)
    threshold = float(args.threshold)
    num_queries = int(args.num_queries)

    # Pad to multiple of crop_size
    img_pil_padded, (pad_h, pad_w) = pad_to_multiple_pil(img_pil, crop_size)
    
    # Apply the same transforms as the dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    patches, num_h, num_w, H, W = split_into_patches_pil(img_pil_padded, crop_size, transform)
    patches = patches.cuda()
    
    num_patches = patches.shape[0]
    print(f"\nImage: {img_pil.size} (W x H)")
    print(f"Padded: {img_pil_padded.size}")
    print(f"Patches: {num_patches} ({num_h}x{num_w})")
    print(f"DEBUG: patches.shape = {patches.shape}")
    print(f"DEBUG: patches min/max = {patches.min():.4f}/{patches.max():.4f}")

    outputs = model(patches)
    real_density_map = None
    
    if isinstance(outputs, list) and len(outputs) == 2:
        out_dict, out_dm = outputs
        # Density map processing
        patch_density_maps = out_dm[1]
        N, C, h, w = patch_density_maps.shape
        patch_density_maps = patch_density_maps.view(num_h, num_w, C, h, w)
        patch_density_maps = patch_density_maps.permute(2, 0, 3, 1, 4).contiguous()
        full_density_map = patch_density_maps.view(C, num_h * h, num_w * w)[0]
        full_density_map = full_density_map[:H, :W]
        real_density_map = full_density_map.cpu().numpy()
    else:
        out_dict = outputs

    if not isinstance(out_dict, dict):
        raise RuntimeError(f"Unexpected output type: {type(out_dict)}")

    out_logits, out_point = out_dict["pred_logits"], out_dict["pred_points"]
    
    print(f"out_logits: {out_logits.shape}")
    print(f"out_point: {out_point.shape}")
    print(f"DEBUG: out_logits min/max = {out_logits.min():.4f}/{out_logits.max():.4f}")

    # ========== MATCH TEST.PY EXACTLY ==========
    prob = out_logits.sigmoid()
    prob = prob.view(1, -1, 2)  # [1, num_patches*num_queries, 2]
    out_logits_reshaped = out_logits.view(1, -1, 2)
    
    topk_k = num_patches * num_queries
    
    topk_values, topk_indexes = torch.topk(
        prob.view(out_logits_reshaped.shape[0], -1),  # [1, 56000]
        topk_k,
        dim=1
    )
    
    print(f"topk_values shape: {topk_values.shape}")
    print(f"topk_k: {topk_k}")
    print(f"DEBUG: topk_values min/max = {topk_values.min():.4f}/{topk_values.max():.4f}")
    print(f"Predictions above threshold: {(topk_values > threshold).sum().item()}")
    
    # Match test.py counting logic
    count = 0
    for k in range(topk_values.shape[0]):
        sub_count = topk_values[k, :]
        sub_count = sub_count.clone()
        sub_count[sub_count < threshold] = 0
        sub_count[sub_count > 0] = 1
        sub_count = torch.sum(sub_count).item()
        count += sub_count
    
    print(f"\nFinal count from logits: {count}")
    
    # If dm_count is enabled
    if args.dm_count and isinstance(outputs, list) and len(outputs) == 2:
        count_dm = 0
        for k in range(out_dm[1].shape[0]):
            count_dm += out_dm[1][k,:].sum().item()
        print(f"Count from density map: {count_dm}")
        print(f"Average: {(count + count_dm) / 2}")
    
    # ========== VISUALIZATION MATCHING TEST.PY ==========
    # For visualization, we need to properly map points to patches
    # The topk_indexes are from flattened [batch, num_patches*num_queries*2]
    # We need to map back to [patch_idx, query_idx]
    
    # topk_indexes is in range [0, 56000) for 40 patches * 700 queries * 2 classes
    # Each query has 2 class predictions, so:
    # index = patch_idx * num_queries * 2 + query_idx * 2 + class_idx
    
    topk_points_idx = topk_indexes // 2  # Which query (ignoring class)
    topk_class = topk_indexes % 2        # Which class (0 or 1)
    
    # Now map to actual points
    # out_point shape: [num_patches, num_queries, 3] where 3 = [x, y, ?]
    # We need the first 2 dimensions [x, y]
    out_point_xy = out_point[:, :, :2]  # [num_patches, num_queries, 2]
    
    # Flatten to [num_patches * num_queries, 2]
    out_point_flat = out_point_xy.reshape(-1, 2)
    
    # Select points based on topk indices
    out_point_selected = out_point_flat[topk_points_idx[0]]  # [28000, 2]
    out_point_selected = out_point_selected * crop_size  # Scale to pixel coordinates
    
    # Reconstruct for visualization - distribute to patches
    value_points = torch.zeros(num_patches, num_queries, 3).cuda()
    
    for i in range(topk_k):
        if topk_values[0, i] < threshold:
            continue
        
        # Map flat index back to patch and query
        flat_idx = topk_points_idx[0, i].item()
        patch_idx = flat_idx // num_queries
        query_idx = flat_idx % num_queries
        
        if patch_idx < num_patches and query_idx < num_queries:
            value_points[patch_idx, query_idx, 0] = topk_values[0, i]
            value_points[patch_idx, query_idx, 1:] = out_point_selected[i]
    
    value_points = value_points.unsqueeze(1)  # [num_patches, 1, num_queries, 3]
    # ========== END MATCHING LOGIC ==========

    # Convert padded PIL image to BGR numpy for visualization
    img_padded_np = np.array(img_pil_padded)
    img_padded_bgr = cv2.cvtColor(img_padded_np, cv2.COLOR_RGB2BGR)
    
    point_map, density_map, drawn, count_visual = show_map(
        value_points, img_padded_bgr.copy(), W, H, crop_size, num_h, num_w, threshold=threshold
    )
    
    print(f"Count from visualization: {count_visual}\n")

    # Annotate
    drawn_vis = drawn.copy()
    cv2.putText(drawn_vis, f"Count: {int(count)}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Crop back to original size if padded
    ow, oh = img_pil.size
    if pad_h > 0 or pad_w > 0:
        drawn_vis = drawn_vis[:oh, :ow]
        point_map = point_map[:oh, :ow]
        density_map = density_map[:oh, :ow]
        if real_density_map is not None:
            real_density_map = real_density_map[:oh, :ow]

    out_points_path = os.path.join(out_dir, f"{base}_pred_points.png")
    out_pointmap_path = os.path.join(out_dir, f"{base}_point_map.png")
    out_density_path = os.path.join(out_dir, f"{base}_density_map.png")

    cv2.imwrite(out_points_path, drawn_vis)
    cv2.imwrite(out_pointmap_path, point_map)
    cv2.imwrite(out_density_path, density_map)

    # Save real density map
    if real_density_map is not None:
        dm = real_density_map
        if dm.shape[0] != oh or dm.shape[1] != ow:
            dm = cv2.resize(dm, (ow, oh), interpolation=cv2.INTER_CUBIC)
        if np.max(dm) > 0:
            dm = dm / np.max(dm) * 255
        dm = dm.astype(np.uint8)
        dm_color = cv2.applyColorMap(dm, 2)
        out_real_density_path = os.path.join(out_dir, f"{base}_real_density_map.png")
        cv2.imwrite(out_real_density_path, dm_color)
        print("  ", out_real_density_path)

    # --- GRID & MASK OUTPUT (behave like test_patch_grid_output.py) ---
    # Configurable via args if present, otherwise use defaults
    grid_rows = getattr(args, 'grid_rows', 8)
    grid_cols = getattr(args, 'grid_cols', 8)
    dense_thr = getattr(args, 'dense_thr', 2.0)
    mask_mode = getattr(args, 'mask_mode', 'blur')

    # Use the real density map if available to compute grid counts and masking
    try:
        if real_density_map is not None:
            dm_for_grid = real_density_map
            if dm_for_grid.shape[0] != oh or dm_for_grid.shape[1] != ow:
                dm_for_grid = cv2.resize(dm_for_grid, (ow, oh), interpolation=cv2.INTER_CUBIC)
            # compute cell counts
            pred_counts = grid_counts_from_density(dm_for_grid, grid_rows, grid_cols)
            dense_mask_cells = pred_counts >= dense_thr

            # Create masked image (operate on cropped original-size image)
            img_for_mask = img_padded_bgr.copy()[:oh, :ow]
            masked_img = mask_image_by_cells(img_for_mask, dense_mask_cells, grid_rows, grid_cols, dm_for_grid.shape, mode=mask_mode)

            # Draw grid lines over masked image for clarity
            masked_img_with_grid = draw_grid_lines(masked_img.copy(), grid_rows, grid_cols, color=(0, 255, 0), thickness=2)

            out_masked_path = os.path.join(out_dir, f"{base}_masked_grid.png")
            cv2.imwrite(out_masked_path, masked_img_with_grid)
            print("  ", out_masked_path)
        else:
            # If no density map available, still save a grid overlay on the prediction visualization
            grid_overlay = draw_grid_lines(drawn_vis.copy(), grid_rows, grid_cols, color=(0, 255, 0), thickness=2)
            out_grid_path = os.path.join(out_dir, f"{base}_grid_overlay.png")
            cv2.imwrite(out_grid_path, grid_overlay)
            print("  ", out_grid_path)
    except Exception as e:
        print("Warning: grid/mask generation failed:", e)

    print("Saved:")
    print("  ", out_points_path)
    print("  ", out_pointmap_path)
    print("  ", out_density_path)
    print(f"Predicted count: {int(count)}")


def main():
    model = build_and_load_model()
    infer_and_save_single_image(model)


if __name__ == "__main__":
    main()