#!/bin/bash
# Setup script for Medical-SigLIP project

echo "========================================"
echo "Medical-SigLIP Setup"
echo "========================================"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Create necessary directories
echo "Creating output directories..."
mkdir -p checkpoints outputs/splits outputs/logs outputs/evaluation outputs/qualitative outputs/comparison

# Prepare data splits
echo "Preparing data splits..."
python main.py prepare_data

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Train SigLIP: python main.py train_siglip --peft_method lora"
echo "3. See README.md for more details"
echo ""

