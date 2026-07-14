================================================================================
PROCEDURAL UV MAP MOLD AUGMENTATION - DOCUMENTATION
================================================================================

This Python script is designed to automatically generate realistic, organic 
mold onto a 3D model's UV Texture Map. Instead of pasting mold, it mathematically
"grows" the infection to make it look natural.

Then it outputs a black-and-white "Kernel Mask" (Ground Truth) 
showing the exact pixel location of the mold.

--------------------------------------------------------------------------------
HOW THE CODE WORKS 
--------------------------------------------------------------------------------

1. DETECTING THE UV ISLANDS (Finding the Cauliflower)
   A UV Map consists of 2D pieces of the 3D model scattered across a pure black 
   background. The script analyzes the image and creates a binary mask. Anything 
   brighter than black (threshold value 5) is marked as "the plant", 
   telling the code exactly where it is allowed to paint.

2. SEAM PROTECTION (Preventing Cut-off Mold in 3D)
   If a mold spot lands right on the edge of a UV island, it will be   
   sliced in half when you wrap it back onto the 3D model. 
   To fix this, the script erodes (shrinks) the valid texture area using 
   a large 50x50 pixel brush. The script is forced to pick random coordinates 
   ONLY within this shrunken "safe zone". This guarantees the mold always grows 
   deep inside the texture pieces and never hits a 3D cut line.

3. CREATING THE ORGANIC AMOEBA (The Blob Shape)
   To prevent predictable geometric shapes, the script generates a unique 
   irregular silhouette for every single spot:
   - It picks a random center point inside the safe zone.
   - It stamps 4 to 8 overlapping circles with randomized sizes 
     and small directional offsets.
   - It applies a heavy Gaussian Blur to melt these circles into a fluid, organic mass.
   - It applies a threshold to solidify the shape, followed by a lighter blur
     to keep the outer edges soft and naturally faded.

4. MOLD TEXTURE PREPARATION (Color and Brightness Tuning)
   The script takes your original 'blackMold.png' texture and prepares it:
   - It resizes and crops it randomly so that no two mold spots ever share the 
     same pattern or resolution.
   - It temporarily converts the patch into the HSV color space 
     and multiplies the Brightness by a random factor between 1.5 and 2.2. 
     This prevents the mold from turning into pitch-black.

5. FINAL COMPOSITING (Mathematical Blending)
   The algorithm blends the mold into the cauliflower texture pixel-by-pixel 
   using standard alpha compositing formulas:
   
   Final Pixel Color = (Original Texture * (1 - Alpha)) + (Mold Texture * Alpha)
   
   The Alpha opacity value is determined by multiplying the procedural amoeba shape, 
   the mold's native transparency channel, and a random visibility factor (50% to 90%). 
   Finally, it clips the result against the original UV island mask to ensure 
   absolute zero bleeding into the black background.

6. KERNEL MASK GENERATION (Ground Truth for AI)
   While painting the texture, the script updates a completely black image canvas 
   of the exact same size. Every pixel where the mold ends up being clearly visible 
   (opacity > 15%) is stamped with pure white using a bitwise OR operation. 
   This outputs the 'kernel_mask.png', giving your neural network a perfect pixel-level 
   map of the defect.

--------------------------------------------------------------------------------
COMMAND LINE USAGE
--------------------------------------------------------------------------------

Basic Command:
python mold.py -i <texture.jpg> -m <mold.png>

Required Arguments:
  -i, --uvmap       Path to your clean target UV texture map image.
  -m, --mold        Path to your black mold/fungus asset source.

Optional Arguments:
  -d, --dir         Define the destination directory where all outputs will be saved.
                    If the folder does not exist, the script will create it automatically.
                    (Default: Saves in the current working directory)
  -o, --output_uv   Define a custom filename for the infected texture (Default: uv_mold.jpg)
  -k, --kernel      Define a custom filename for the B&W kernel mask (Default: kernel_mask.png)
  -t, --total_molds Set the total number of mold spots to scatter (Default: 10)

Example Usage:
python mold.py -i cauliflower/new/extracted/image0.jpg -m blackMold.png -d cauliflower/new/Result -t 12
