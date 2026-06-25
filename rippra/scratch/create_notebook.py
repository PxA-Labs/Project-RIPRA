# scratch/create_notebook.py - Helper script to generate valid .ipynb file using Base64 encoding
import json
import os
import base64

def main():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # 1. Title cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Project RIPRA (ऋप्र): Shack-Hartmann Wavefront Reconstruction & Turbulence Characterization\n",
            "This notebook runs the data generation, model definition, and training pipeline in Kaggle/Colab with GPU acceleration.\n",
            "\n",
            "### Objectives:\n",
            "1. Generate a massive synthetic Kolmogorov dataset based on physical Shack-Hartmann parameters.\n",
            "2. Train a Fully Connected Network (MLP) as a baseline.\n",
            "3. Train a Spatial 2D ResNet CNN mapping displacements to Zernike coefficients."
        ]
    })

    # 2. Setup cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 1: Write All Configuration, Spot Coordinates, and Python Files to Disk\n",
            "We write all configuration files and scripts directly to the Kaggle filesystem via Base64 decoding (to avoid quote/triple-quote nesting syntax errors)."
        ]
    })

    # Read and base64-encode all files
    def get_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    conf_b64 = get_b64("config/system.conf")
    spots_b64 = get_b64("results/reference_centroids_c.csv")
    gen_b64 = get_b64("tools/generate_dataset.py")
    models_b64 = get_b64("ml/models.py")
    train_b64 = get_b64("ml/train.py")
    seq_models_b64 = get_b64("ml/sequence_models.py")
    train_seq_b64 = get_b64("ml/train_sequence.py")

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import base64\n",
            "\n",
            "# Create required directory structures\n",
            "os.makedirs('config', exist_ok=True)\n",
            "os.makedirs('results', exist_ok=True)\n",
            "os.makedirs('data_ai', exist_ok=True)\n",
            "os.makedirs('ml_checkpoints', exist_ok=True)\n",
            "os.makedirs('ml_checkpoints/kaggle', exist_ok=True)\n",
            "\n",
            f"conf_b64 = '{conf_b64}'\n",
            "with open('config/system.conf', 'wb') as f:\n",
            "    f.write(base64.b64decode(conf_b64))\n",
            "\n",
            f"spots_b64 = '{spots_b64}'\n",
            "with open('results/reference_centroids_c.csv', 'wb') as f:\n",
            "    f.write(base64.b64decode(spots_b64))\n",
            "\n",
            f"gen_b64 = '{gen_b64}'\n",
            "with open('generate_dataset.py', 'wb') as f:\n",
            "    f.write(base64.b64decode(gen_b64))\n",
            "\n",
            f"models_b64 = '{models_b64}'\n",
            "with open('models.py', 'wb') as f:\n",
            "    f.write(base64.b64decode(models_b64))\n",
            "\n",
            f"train_b64 = '{train_b64}'\n",
            "with open('train.py', 'wb') as f:\n",
            "    f.write(base64.b64decode(train_b64))\n",
            "\n",
            f"seq_models_b64 = '{seq_models_b64}'\n",
            "with open('sequence_models.py', 'wb') as f:\n",
            "    f.write(base64.b64decode(seq_models_b64))\n",
            "\n",
            f"train_seq_b64 = '{train_seq_b64}'\n",
            "with open('train_sequence.py', 'wb') as f:\n",
            "    f.write(base64.b64decode(train_seq_b64))\n",
            "\n",
            "print('All files decoded and written to disk successfully!')"
        ]
    })

    # 3. Generate Data cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2: Generate Massive Dataset\n",
            "Generate 50,000 samples of temporally correlated synthetic Kolmogorov turbulence."
        ]
    })

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python generate_dataset.py --samples 50000 --out data_ai/dataset.npz --noise 0.1"
        ]
    })

    # 4. Train MLP cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Train WavefrontMLP Reconstructor\n",
            "Train the MLP baseline for 30 epochs."
        ]
    })

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python train.py --model mlp --epochs 30 --batch_size 128 --lr 1e-3"
        ]
    })

    # 5. Train CNN cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Train WavefrontCNN Reconstructor\n",
            "Train the ResNet-based CNN reconstructor for 30 epochs."
        ]
    })

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python train.py --model cnn --epochs 30 --batch_size 128 --lr 1e-3"
        ]
    })

    # 6. Train LSTM Wavefront Predictor
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Train LSTM Wavefront Predictor (Phase 5 - Checkpoint 5.1)\n",
            "Train the LSTM to predict future Zernike coefficients (lookback=10 frames, prediction step=1 frame)."
        ]
    })

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python train_sequence.py --task predict --epochs 20 --batch_size 128 --lookback 10 --step 1"
        ]
    })

    # 7. Train LSTM Turbulence Classifier
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Train LSTM Turbulence Classifier (Phase 5 - Checkpoint 5.2)\n",
            "Train the LSTM to classify sequences into Weak, Moderate, or Strong turbulence regimes."
        ]
    })

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python train_sequence.py --task classify --epochs 20 --batch_size 128 --lookback 10"
        ]
    })

    # 8. Train LSTM Parameter Estimator
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7: Train LSTM Turbulence Parameter Estimator (Phase 5 - Checkpoint 5.3)\n",
            "Train the LSTM regression network to predict the Fried parameter (D/r_0) directly from sequences."
        ]
    })

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python train_sequence.py --task parameter --epochs 20 --batch_size 128 --lookback 10"
        ]
    })

    # 9. Clickable Download Links cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 8: Clickable Download Links\n",
            "Run this cell to generate download links for all trained checkpoints and datasets."
        ]
    })

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import FileLink, display\n",
            "\n",
            "print(\"Click links to download your files:\")\n",
            "display(FileLink('data_ai/dataset.npz'))\n",
            "display(FileLink('ml_checkpoints/best_mlp.pt'))\n",
            "display(FileLink('ml_checkpoints/best_cnn.pt'))\n",
            "display(FileLink('ml_checkpoints/kaggle/best_sequence_predict.pt'))\n",
            "display(FileLink('ml_checkpoints/kaggle/best_sequence_classify.pt'))\n",
            "display(FileLink('ml_checkpoints/kaggle/best_sequence_parameter.pt'))"
        ]
    })

    # Save to file
    out_path = "../notebook/Kaggle_RIPRA_ML_Pipeline.ipynb"
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=1)
        
    print(f"Generated Kaggle notebook at {out_path} successfully!")

if __name__ == "__main__":
    main()
