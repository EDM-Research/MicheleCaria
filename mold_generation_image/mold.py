import cv2
import numpy as np
import random
import argparse
import os

def augment_uv_map(uv_path, mold_path, output_dir, output_uv_name, output_mask_name, total_molds=8):
    # 1. Load the UV Map and the mold texture
    img = cv2.imread(uv_path)
    mold_orig = cv2.imread(mold_path, cv2.IMREAD_UNCHANGED)
    
    if img is None or mold_orig is None:
        print("Error: Could not load images. Please check the file paths.")
        return

    img_h, img_w = img.shape[:2]
    full_ground_truth = np.zeros((img_h, img_w), dtype=np.uint8)

    # 2. Detect UV Map Islands
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, uv_islands_mask = cv2.threshold(gray_img, 5, 255, cv2.THRESH_BINARY)

    # 3. SEAM PROTECTION & AREA RESTRICTION
    kernel_safe = np.ones((50, 50), np.uint8)
    uv_safe_zone = cv2.erode(uv_islands_mask, kernel_safe, iterations=1)
    
    # UNWANTED TOP AREAS ---
    # Dynamically calculate the top 35% of the image and set it to black (0).
    # This completely hides the top-left fragments and the top-right striped block.
    cutoff_y = int(img_h * 0.35)
    uv_safe_zone[0:cutoff_y, :] = 0
    # -------------------------------------
    
    valid_y, valid_x = np.where(uv_safe_zone == 255)

    if len(valid_y) == 0:
        print("Error: The UV Map islands do not have large enough safe areas. Reduce the kernel_safe size.")
        return

    # 4. Mold Generation Loop
    for m in range(total_molds):
        idx = random.randint(0, len(valid_y) - 1)
        cx, cy = valid_x[idx], valid_y[idx]

        S = 300 
        x1, y1 = max(0, cx - S//2), max(0, cy - S//2)
        x2, y2 = min(img_w, cx + S//2), min(img_h, cy + S//2)
        
        patch_w, patch_h = x2 - x1, y2 - y1
        local_cx, local_cy = cx - x1, cy - y1

        # Organic Amoeba Blob Creation
        temp_blob = np.zeros((patch_h, patch_w), dtype=np.uint8)
        num_circles = random.randint(4, 8)
        for _ in range(num_circles):
            ox = random.randint(-30, 30)
            oy = random.randint(-30, 30)
            r = random.randint(15, 40)
            cv2.circle(temp_blob, (local_cx + ox, local_cy + oy), r, 255, -1)
            
        temp_blob = cv2.GaussianBlur(temp_blob, (45, 45), 0)
        _, hard_blob = cv2.threshold(temp_blob, 110, 255, cv2.THRESH_BINARY)
        ameba_alpha = cv2.GaussianBlur(hard_blob, (15, 15), 0).astype(float) / 255.0

        # Mold Texture Scaling and Cropping
        mh, mw = mold_orig.shape[:2]
        scale = max(patch_h / mh, patch_w / mw) * random.uniform(1.0, 1.5)
        new_w, new_h = int(mw * scale), int(mh * scale)
        resized_mold = cv2.resize(mold_orig, (new_w, new_h))
        
        sy = random.randint(0, new_h - patch_h) if new_h > patch_h else 0
        sx = random.randint(0, new_w - patch_w) if new_w > patch_w else 0
        mold_patch = resized_mold[sy:sy+patch_h, sx:sx+patch_w]
        
        if mold_patch.shape[2] == 4:
            patch_rgb = mold_patch[:, :, :3].astype(float)
            patch_tex_alpha = mold_patch[:, :, 3].astype(float) / 255.0
        else:
            patch_rgb = mold_patch[:, :, :3].astype(float)
            patch_tex_alpha = np.ones((patch_h, patch_w), dtype=float)

        # Brightness Correction / Tuning
        hsv = cv2.cvtColor(mold_patch[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(1.5, 2.2), 0, 255)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.8, 0, 255) 
        patch_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(float)

        # Alpha Compositing
        final_alpha = ameba_alpha * patch_tex_alpha * random.uniform(0.5, 0.9)
        island_patch_alpha = cv2.GaussianBlur(uv_islands_mask[y1:y2, x1:x2], (5, 5), 0).astype(float) / 255.0
        final_alpha = final_alpha * island_patch_alpha
        
        final_alpha_3 = np.repeat(final_alpha[:, :, np.newaxis], 3, axis=2)
        img_patch = img[y1:y2, x1:x2].astype(float)
        blended = img_patch * (1.0 - final_alpha_3) + patch_rgb * final_alpha_3
        
        img[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        # Update Ground Truth Mask
        visible_mask = (final_alpha > 0.15).astype(np.uint8) * 255
        patch_gt = full_ground_truth[y1:y2, x1:x2]
        full_ground_truth[y1:y2, x1:x2] = cv2.bitwise_or(patch_gt, visible_mask)

    # 5. Save Outputs
    os.makedirs(output_dir, exist_ok=True)
    
    full_output_uv_path = os.path.join(output_dir, output_uv_name)
    full_output_mask_path = os.path.join(output_dir, output_mask_name)
    
    cv2.imwrite(full_output_uv_path, img)
    cv2.imwrite(full_output_mask_path, full_ground_truth)
    
    print(f"Dataset Generated! Files successfully saved into: {output_dir}")

# --- EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply organic mold directly onto a 3D model's UV Texture Map.")
    
    parser.add_argument("-i", "--uvmap", required=True, help="Path to original clean UV Map texture.")
    parser.add_argument("-m", "--mold", required=True, help="Path to alpha mold texture.")
    parser.add_argument("-d", "--dir", default="mold_dataset_outputs", help="Target directory where outputs will be saved.")
    parser.add_argument("-o", "--output_uv", default="uv_mold.jpg", help="Filename for the modified UV map.")
    parser.add_argument("-k", "--kernel", default="kernel_mask.png", help="Filename for the B&W ground truth mask.")
    parser.add_argument("-t", "--total_molds", type=int, default=10, help="Total mold spots to inject.")

    args = parser.parse_args()

    augment_uv_map(
        uv_path=args.uvmap,
        mold_path=args.mold,
        output_dir=args.dir,
        output_uv_name=args.output_uv,
        output_mask_name=args.kernel,
        total_molds=args.total_molds
    )