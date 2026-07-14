import cv2
import os
import sys

class InteractiveCropTool:
    def __init__(self, image_path, output_dir="my_custom_dataset", label_id=0):
        self.image_path = image_path
        self.label_id = label_id
        
        # Load the image
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Cannot find or load the image: {image_path}")
        
        self.clone = self.img.copy()
        self.h, self.w = self.img.shape[:2]
        
        # Create output directories in the current working directory
        current_dir = os.getcwd()
        full_output_path = os.path.join(current_dir, output_dir)
        
        self.images_dir = os.path.join(full_output_path, "images")
        self.labels_dir = os.path.join(full_output_path, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # State variables for drawing
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.bbox = None
        self.current_crop_coords = None
        
        # Count existing files to avoid overwriting
        self.saved_count = len([name for name in os.listdir(self.images_dir) if name.endswith('.jpg')])
        
        # Initialize the window
        self.window_name = "Cropping Tool (512x512)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.draw_and_calculate)

    def draw_and_calculate(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_img = self.clone.copy()
                cv2.rectangle(temp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox = (min(self.ix, x), min(self.iy, y), max(self.ix, x), max(self.iy, y))
            temp_img = self.clone.copy()
            
            cx = (self.bbox[0] + self.bbox[2]) // 2
            cy = (self.bbox[1] + self.bbox[3]) // 2
            
            x1_crop = max(0, cx - 256)
            y1_crop = max(0, cy - 256)
            x2_crop = x1_crop + 512
            y2_crop = y1_crop + 512
            
            if x2_crop > self.w:
                x2_crop = self.w
                x1_crop = max(0, x2_crop - 512)
            if y2_crop > self.h:
                y2_crop = self.h
                y1_crop = max(0, y2_crop - 512)
            
            cv2.rectangle(temp_img, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (0, 255, 0), 2)
            cv2.rectangle(temp_img, (x1_crop, y1_crop), (x2_crop, y2_crop), (255, 0, 0), 2)
            cv2.imshow(self.window_name, temp_img)
            self.current_crop_coords = (x1_crop, y1_crop, x2_crop, y2_crop)

    def save_crop(self):
        if not self.current_crop_coords or not self.bbox:
            return

        x1_crop, y1_crop, x2_crop, y2_crop = self.current_crop_coords
        bx1, by1, bx2, by2 = self.bbox
        
        crop_img = self.img[y1_crop:y2_crop, x1_crop:x2_crop]
        
        new_bx1 = max(0, bx1 - x1_crop)
        new_by1 = max(0, by1 - y1_crop)
        new_bx2 = min(512, bx2 - x1_crop)
        new_by2 = min(512, by2 - y1_crop)
        
        b_width = new_bx2 - new_bx1
        b_height = new_by2 - new_by1
        b_cx = new_bx1 + (b_width / 2.0)
        b_cy = new_by1 + (b_height / 2.0)
        
        yolo_x = b_cx / 512.0
        yolo_y = b_cy / 512.0
        yolo_w = b_width / 512.0
        yolo_h = b_height / 512.0
        
        self.saved_count += 1
        base_filename = f"black_mold_{self.saved_count:04d}"
        img_path = os.path.join(self.images_dir, f"{base_filename}.jpg")
        label_path = os.path.join(self.labels_dir, f"{base_filename}.txt")
        
        cv2.imwrite(img_path, crop_img)
        
        with open(label_path, "w") as f:
            f.write(f"{self.label_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
            
        print(f"Saved successfully: {base_filename}")
        self.clone = self.img.copy()
        self.bbox = None
        cv2.imshow(self.window_name, self.clone)

    def run(self):
        print("\n--- TOOL INSTRUCTIONS ---")
        print("1. Click and drag to draw a rectangle (green) tightly around the mold.")
        print("2. The blue box shows the 512x512 area that will be saved.")
        print("3. Press 's' to save, 'r' to reset, 'q' to quit.\n")
        
        cv2.imshow(self.window_name, self.clone)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.clone = self.img.copy()
                self.bbox = None
                cv2.imshow(self.window_name, self.clone)
            elif key == ord("s"):
                self.save_crop()
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_image>")
        sys.exit(1)
        
    img_arg = sys.argv[1]
    tool = InteractiveCropTool(image_path=img_arg, output_dir='cropping', label_id=0)
    tool.run()