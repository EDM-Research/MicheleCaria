# Pipeline Overview: From 2D Image to Mold-Augmented 3D Model

This document provides a high-level summary of our workflow for generating 3D models and procedurally applying synthetic mold defects.

## 1. The Pipeline Workflow
1. **Input Image:** The process begins with a 2D source image (either a real photograph or an AI-generated picture).
2. **3D Generation:** The image is fed into our SPAR3D web interface. The model reconstructs the geometry and outputs a complete 3D mesh (.glb).
3. **Texture Extraction:** A texture extractor isolates and pulls the raw 2D UV texture map (e.g., `image0.jpg`) directly from the generated 3D model.
4. **Mold Augmentation:** The extracted texture is passed to our custom script (`mold.py`), which procedurally "grows" realistic mold onto the texture and generates a Ground Truth mask for AI training.
5. **Blender try** You should use only the real model on blender and then apply the new texture as a material, the 3D model that u get after the extraction wont work the texture wont apply.
## 2. Running the 3D Generator (Gradio)
To start the local web interface for the 3D generation step, open your terminal inside your virtual environment and run:

```bash
python gradio_app.py
