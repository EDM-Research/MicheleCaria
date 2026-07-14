#!/bin/bash

echo "Creating the Conda environment..."
conda create -y -n my_env python=3.10

echo "Initializing Conda for the script..."
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Activating the environment..."
conda activate my_env

# --- AGGIUNGI QUESTO COMANDO QUI ---
echo "Installing local CUDA toolkit to match PyTorch..."
conda install -y -c nvidia cuda-toolkit=13.0
# -----------------------------------

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing standard dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing custom 3D modules..."
pip install --no-build-isolation ./stable-point-aware-3d/uv_unwrapper
pip install --no-build-isolation ./stable-point-aware-3d/texture_baker

echo "============================================================"
echo "Installation complete!"
echo "To start working in your terminal, type:"
echo "conda activate my_env"
echo "============================================================"
