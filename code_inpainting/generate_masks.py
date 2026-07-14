from __future__ import annotations
import cv2
import numpy as np
import os
import argparse
import shutil
from pathlib import Path

# Global variables for interface state management
current_img = None
current_mask = None
mask_history = []  
brush_size = 5     
drawing = False    

def draw_mask(event, x, y, flags, param):
    global current_mask, mask_history, brush_size, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mask_history.append(current_mask.copy())
        cv2.circle(current_mask, (x, y), brush_size, 255, -1)
        update_view()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(current_mask, (x, y), brush_size, 255, -1)
            update_view()

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(current_mask, (x, y), brush_size, 255, -1)
        update_view()

def update_view():
    mask_bgr = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
    color_layer = np.zeros_like(current_img)
    color_layer[current_mask == 255] = [0, 255, 0]
    overlay = cv2.addWeighted(current_img, 1.0, color_layer, 0.4, 0)
    
    contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    
    cv2.circle(overlay, (30, 30), brush_size, (255, 255, 255), 1)
    cv2.putText(overlay, f"Brush: {brush_size}", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    combined = np.hstack((overlay, mask_bgr))
    cv2.imshow("Annotation Tool - Manual Brush", combined)

def process_image(img_path: Path, result_dir: Path):
    global current_img, current_mask, mask_history
    
    # Set up result subdirectories
    img_out_dir = result_dir / "images"
    mask_out_dir = result_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    
    mask_path = mask_out_dir / f"{img_path.stem}_mask{img_path.suffix}"
    img_copy_path = img_out_dir / img_path.name

    current_img = cv2.imread(str(img_path))
    if current_img is None:
        print(f"Skipping: Could not load {img_path}")
        return False

    h, w = current_img.shape[:2]
    current_mask = np.zeros((h, w), dtype=np.uint8)
    mask_history = []

    # Preload existing mask if present
    if mask_path.exists():
        loaded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if loaded is not None:
            current_mask = loaded

    print(f"Annotating: {img_path.name}")
    update_view()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save mask and copy original image
            cv2.imwrite(str(mask_path), current_mask)
            shutil.copy2(str(img_path), str(img_copy_path))
            print(f"Saved: {mask_path.name} and {img_copy_path.name}")
            return True
        elif key == ord('z'):
            if mask_history:
                current_mask = mask_history.pop()
                update_view()
        elif key in [ord('+'), ord('=')]:
            global brush_size
            brush_size += 1
            update_view()
        elif key == ord('-'):
            brush_size = max(1, brush_size - 1)
            update_view()
        elif key == 27:
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Manual Mask Annotation Tool")
    parser.add_argument("path", help="Path to an image file or a directory of images")
    args = parser.parse_args()

    target_path = Path(args.path)
    
    # Determine the result directory based on input location
    if target_path.is_file():
        image_list = [target_path]
        result_dir = target_path.parent / "result"
    elif target_path.is_dir():
        image_list = [f for f in target_path.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        result_dir = target_path / "result"
    else:
        print("Error: Path not found.")
        return

    cv2.namedWindow("Annotation Tool - Manual Brush", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotation Tool - Manual Brush", draw_mask)

    for img_path in sorted(image_list):
        if not process_image(img_path, result_dir):
            break

    cv2.destroyAllWindows()
    print("\nAnnotation session closed.")

if __name__ == "__main__":
    main()
