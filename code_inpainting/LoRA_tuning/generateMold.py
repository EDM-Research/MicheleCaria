from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel
from PIL import Image, ImageFilter

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
LORA_PATH = "./black_mold_lora_out/final_lora"

OUTPUT_DIR = Path("./generated_black_mold")
OUT_NAME = "final_highres_mold.png"
MASK_OUT_NAME = "final_mask_512.png"

PROMPT = "realistic black mold on organic surface, dark fungus, mold spots"
NEGATIVE_PROMPT = (
    "crack, scratch, line, linear, hairline, split, "
    "green tint, blue spots, red marks, colorful artifacts, painted blob, "
    "cartoon, illustration, glossy, plastic, watermark, text, "
    "low quality, blurry, oversaturated, sharp edges, clean"
)

# Global variables for UI state
crop_x, crop_y = 0, 0
roi_img = None
roi_mask = None
mask_history = []
brush_size = 5
drawing = False
step = 0  # 0: Crop Selection, 1: Brush Drawing

def interactive_callback(event, x, y, flags, param):
    global crop_x, crop_y, roi_img, roi_mask, mask_history, brush_size, drawing, step, current_frame
    
    # STEP 0: Move a fixed 512x512 square overlay
    if step == 0:
        if event == cv2.EVENT_MOUSEMOVE:
            current_frame = param.copy()
            crop_x = max(0, min(x - 256, param.shape[1] - 512))
            crop_y = max(0, min(y - 256, param.shape[0] - 512))
            cv2.rectangle(current_frame, (crop_x, crop_y), (crop_x + 512, crop_y + 512), (0, 255, 0), 2)
            cv2.putText(current_frame, "Click to lock 512x512 region", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Interactive Pipeline", current_frame)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            roi_img = param[crop_y:crop_y+512, crop_x:crop_x+512].copy()
            roi_mask = np.zeros((512, 512), dtype=np.uint8)
            step = 1
            update_brush_view()

    # STEP 1: Manual brush masking inside the locked ROI
    elif step == 1:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            mask_history.append(roi_mask.copy())
            cv2.circle(roi_mask, (x, y), brush_size, 255, -1)
            update_brush_view()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(roi_mask, (x, y), brush_size, 255, -1)
            update_brush_view()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

def update_brush_view():
    global roi_img, roi_mask, brush_size
    mask_bgr = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    color_layer = np.zeros_like(roi_img)
    color_layer[roi_mask == 255] = [0, 255, 0]
    overlay = cv2.addWeighted(roi_img, 1.0, color_layer, 0.4, 0)
    
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    
    cv2.circle(overlay, (30, 30), brush_size, (255, 255, 255), 1)
    cv2.putText(overlay, f"Brush: {brush_size} | Press 'S' to Inpaint", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    combined = np.hstack((overlay, mask_bgr))
    cv2.imshow("Interactive Pipeline", combined)

def load_pipe(device: str, dtype: torch.dtype) -> StableDiffusionInpaintPipeline:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype, safety_checker=None).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
    pipe.unet.to(device=device, dtype=dtype).eval()
    return pipe

def main():
    global step, roi_img, roi_mask, mask_history, brush_size, current_frame
    
    parser = argparse.ArgumentParser(description="Interactive high-resolution inpainting pipeline.")
    parser.add_argument("--image", type=str, required=True, help="Path to the target input image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Target file '{image_path}' does not exist.")
        return

    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        print(f"Error: Failed to load image file from '{image_path}'.")
        return

    h, w = orig_bgr.shape[:2]
    if h < 512 or w < 512:
        print(f"Error: Source image resolution ({w}x{h}) must be at least 512x512 pixels.")
        return

    current_frame = orig_bgr.copy()
    cv2.namedWindow("Interactive Pipeline", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Interactive Pipeline", interactive_callback, param=orig_bgr)
    
    cv2.imshow("Interactive Pipeline", current_frame)

    while step == 0:
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            break
        elif key == ord('z') or key == ord('Z'):
            if mask_history:
                roi_mask = mask_history.pop()
                update_brush_view()
        elif key == ord('+') or key == ord('='):
            brush_size += 1
            update_brush_view()
        elif key == ord('-'):
            brush_size = max(1, brush_size - 1)
            update_brush_view()
        elif key == 27:
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    
    if roi_mask.max() == 0:
        print("Execution halted: Mask cannot be completely black.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / MASK_OUT_NAME), roi_mask)
    print(f"Mask asset saved successfully to: {OUTPUT_DIR / MASK_OUT_NAME}")

    roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    pil_roi_img = Image.fromarray(roi_img_rgb)
    pil_roi_mask = Image.fromarray(roi_mask).convert("L")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print("Loading generative model pipeline...")
    pipe = load_pipe(device, dtype)

    print("Running localized neural inpainting pass...")
    with torch.inference_mode():
        inpainted_roi = pipe(
            prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
            image=pil_roi_img, mask_image=pil_roi_mask,
            guidance_scale=8.5, num_inference_steps=40,
            width=512, height=512
        ).images[0]

    feathered_mask = pil_roi_mask.filter(ImageFilter.GaussianBlur(radius=4))
    blended_roi_pil = Image.composite(inpainted_roi, pil_roi_img, feathered_mask)
    
    blended_roi_bgr = cv2.cvtColor(np.array(blended_roi_pil), cv2.COLOR_RGB2BGR)
    
    final_output = orig_bgr.copy()
    final_output[crop_y:crop_y+512, crop_x:crop_x+512] = blended_roi_bgr

    output_path = OUTPUT_DIR / OUT_NAME
    cv2.imwrite(str(output_path), final_output)
    print(f"Process complete. Output re-projected successfully to: {output_path}")

    # Display the final generated high-resolution output immediately
    print("\nDisplaying the generated high-resolution image. Press any key to close the window.")
    cv2.namedWindow("Final Output Preview", cv2.WINDOW_NORMAL)
    cv2.imshow("Final Output Preview", final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()