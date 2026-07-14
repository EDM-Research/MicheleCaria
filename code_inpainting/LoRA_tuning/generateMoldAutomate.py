from __future__ import annotations
import argparse
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

# --- CONFIGURATION ---
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
LORA_PATH = "./black_mold_lora_out/final_lora"

PROMPT = "realistic black mold on organic surface, dark fungus, mold spots"
NEGATIVE_PROMPT = (
    "crack, scratch, line, linear, hairline, split, "
    "green tint, blue spots, red marks, colorful artifacts, painted blob, "
    "cartoon, illustration, glossy, plastic, watermark, text, "
    "low quality, blurry, oversaturated, sharp edges, clean"
)

# --- GLOBAL UI STATE ---
drawing = False
ix, iy = -1, -1
zones = []  # Stores defined bounding boxes as (x1, y1, x2, y2)
display_img = None
original_bgr = None

def interactive_draw(event, x, y, flags, param):
    """Handles OpenCV mouse events to let the user draw bounding boxes."""
    global ix, iy, drawing, display_img, original_bgr, zones

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            display_img = original_bgr.copy()
            for (zx1, zy1, zx2, zy2) in zones:
                cv2.rectangle(display_img, (zx1, zy1), (zx2, zy2), (0, 255, 0), 2)
            cv2.rectangle(display_img, (ix, iy), (x, y), (0, 165, 255), 2)
            cv2.imshow("Zone Selector", display_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, x2 = min(ix, x), max(ix, x)
        y1, y2 = min(iy, y), max(iy, y)
        
        # Prevent 1-pixel accidental clicks
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            zones.append((x1, y1, x2, y2))
            
        display_img = original_bgr.copy()
        for (zx1, zy1, zx2, zy2) in zones:
            cv2.rectangle(display_img, (zx1, zy1), (zx2, zy2), (0, 255, 0), 2)
        cv2.imshow("Zone Selector", display_img)

def generate_constrained_mask(w: int, h: int, num_spots: int = 4) -> Image.Image:
    """
    Generates procedural mold spots strictly confined within the given width and height.
    Ellipses will not bleed past the coordinates 0 and w/h.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    
    for _ in range(num_spots):
        # Keep cluster centers slightly padded from the absolute edges
        cx = random.randint(int(w * 0.1), int(w * 0.9))
        cy = random.randint(int(h * 0.1), int(h * 0.9))
        base_radius = random.randint(15, max(20, min(w, h) // 3))
        
        for _ in range(random.randint(3, 6)):
            offset_x = random.randint(-base_radius, base_radius)
            offset_y = random.randint(-base_radius, base_radius)
            r = random.randint(int(base_radius * 0.4), base_radius)
            
            # Strict clamping: forces geometry to stay inside the bounding box
            x0 = max(0, cx + offset_x - r)
            y0 = max(0, cy + offset_y - r)
            x1 = min(w, cx + offset_x + r)
            y1 = min(h, cy + offset_y + r)
            
            if x1 > x0 and y1 > y0:
                draw.ellipse([x0, y0, x1, y1], fill=255)
            
    return mask

def load_inpaint_pipe(device: str, dtype: torch.dtype) -> StableDiffusionInpaintPipeline:
    print("\nLoading generative model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, safety_checker=None
    ).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
    pipe.unet.to(device=device, dtype=dtype).eval()
    return pipe

def main():
    global display_img, original_bgr, zones

    parser = argparse.ArgumentParser(description="Strict Bounding Box Mold Generation.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--output_dir", type=str, default="./automated_dataset", help="Output directory.")
    parser.add_argument("--count", type=int, default=5, help="Total number of images to generate.")
    parser.add_argument("--zones_per_image", type=int, default=1, help="Number of bounding boxes to infect per generated image.")
    args = parser.parse_args()

    input_image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_image_path.exists():
        print(f"Error: File '{input_image_path}' not found.")
        return

    # --- PHASE 1: UI SELECTION ---
    original_bgr = cv2.imread(str(input_image_path))
    if original_bgr is None:
        print(f"Error: Failed to load {input_image_path}")
        return

    display_img = original_bgr.copy()
    
    print("\n" + "="*50)
    print(" INTERACTIVE BOUNDING BOX SELECTOR")
    print("="*50)
    print(" Left Click & Drag : Define bounding boxes.")
    print(" 'z'               : Undo the last box.")
    print(" 's'               : Start generation.")
    print(" 'esc'             : Exit.")
    print("="*50 + "\n")

    cv2.namedWindow("Zone Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Zone Selector", interactive_draw)

    while True:
        cv2.imshow("Zone Selector", display_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') or key == ord('S'):
            if not zones:
                print("Warning: Please draw at least one bounding box.")
                continue
            break
        elif key == ord('z') or key == ord('Z'):
            if zones:
                zones.pop()
                display_img = original_bgr.copy()
                for (zx1, zy1, zx2, zy2) in zones:
                    cv2.rectangle(display_img, (zx1, zy1), (zx2, zy2), (0, 255, 0), 2)
        elif key == 27:
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    print(f"Selection saved. {len(zones)} zones defined.")

    # --- PHASE 2: AUTOMATED GENERATION ---
    original_pil = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    img_w, img_h = original_pil.size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_inpaint_pipe(device, dtype)

    print(f"\nStarting generation of {args.count} images...")

    for i in range(1, args.count + 1):
        print(f"Processing variant {i}/{args.count}...")
        
        current_img = original_pil.copy()
        global_mask = Image.new("L", (img_w, img_h), 0)

        # Randomly select a subset of bounding boxes for this specific iteration
        num_zones = min(args.zones_per_image, len(zones))
        selected_zones = random.sample(zones, num_zones)

        for (x1, y1, x2, y2) in selected_zones:
            box_w, box_h = x2 - x1, y2 - y1
            
            # 1. Generate the local mask exactly the size of the bounding box
            local_mask = generate_constrained_mask(box_w, box_h)

            # 2. Calculate a 512x512 crop area centered around the bounding box for SD
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop_x1 = max(0, min(cx - 256, img_w - 512))
            crop_y1 = max(0, min(cy - 256, img_h - 512))
            crop_x2, crop_y2 = crop_x1 + 512, crop_y1 + 512
            
            base_crop = current_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # 3. Create a 512x512 black mask and paste the local mask into its correct relative position
            sd_mask = Image.new("L", (512, 512), 0)
            paste_x = x1 - crop_x1
            paste_y = y1 - crop_y1
            sd_mask.paste(local_mask, (paste_x, paste_y))

            # 4. Run Neural Inpainting
            with torch.inference_mode():
                inpaint_result = pipe(
                    prompt=PROMPT, 
                    negative_prompt=NEGATIVE_PROMPT,
                    image=base_crop, 
                    mask_image=sd_mask,
                    guidance_scale=8.5, 
                    num_inference_steps=30,
                    width=512, 
                    height=512
                ).images[0]

            # 5. Composite and update the high-res image and the global tracking mask
            feathered_mask = sd_mask.filter(ImageFilter.GaussianBlur(radius=3))
            blended_crop = Image.composite(inpaint_result, base_crop, feathered_mask)

            current_img.paste(blended_crop, (crop_x1, crop_y1))
            global_mask.paste(sd_mask, (crop_x1, crop_y1), sd_mask)

        # 6. Save the final dataset pairs
        file_id = f"dataset_{i:03d}"
        current_img.save(output_dir / f"{file_id}_img.png")
        global_mask.save(output_dir / f"{file_id}_mask.png")
        
    print(f"\nProcess complete. Dataset generated in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()