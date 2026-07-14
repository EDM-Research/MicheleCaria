#!/bin/bash

echo "Creating the Conda environment..."
# We create the environment named 'my_env' using Python 3.12 
# to ensure compatibility with modern library versions.
conda create -y -n my_env python=3.12

echo "Initializing Conda for the script..."
# This command ensures the script knows how to access the 'conda activate' command
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Activating the environment..."
conda activate my_env

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "============================================================"
echo "Installation complete!"
echo "To start working in your terminal, type:"
echo "conda activate my_env"
echo "============================================================"