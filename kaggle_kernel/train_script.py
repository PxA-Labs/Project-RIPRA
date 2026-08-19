import os
import sys
import traceback
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Command '{cmd}' failed with code {result.returncode}\n{result.stderr}")

try:
    print("Setting up Kaggle API...")
    os.system("mkdir -p ~/.kaggle")
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        f.write('{"username":"shlokparekh08","key":"f19f6418812322239ab4bb705464b695"}')
    os.system("chmod 600 ~/.kaggle/kaggle.json")

    print("Downloading dataset...")
    # Download without --unzip so Kaggle CLI doesn't unzip the npz file itself!
    run_cmd("kaggle datasets download -d shlokparekh08/ripra-slope-dataset -p /kaggle/working/")
    
    # Unzip the downloaded zip file which contains dataset_large.npz
    run_cmd("unzip -o /kaggle/working/ripra-slope-dataset.zip -d /kaggle/working/")

    print("Cloning repo...")
    run_cmd("git clone https://github.com/Shlok-Parekh09/Project-RIPRA.git")

    print("Creating dummy reference_centroids_c.csv so export_onnx.py does not crash...")
    os.makedirs("Project-RIPRA/results", exist_ok=True)
    with open("Project-RIPRA/results/reference_centroids_c.csv", "w") as f:
        f.write("x,y\n")
        for i in range(137):
            f.write("0,0\n")

    dataset_path = "/kaggle/working/dataset_large.npz"
    if not os.path.exists(dataset_path):
        run_cmd("ls -R /kaggle/working")
        raise FileNotFoundError(f"Could not find {dataset_path} after download! Is it extracted as individual .npy files?")

    print("Running training...")
    run_cmd(f"cd Project-RIPRA/rippra && CUDA_VISIBLE_DEVICES='' python ml/train_sequence.py --task complete_slopes --dataset {dataset_path} --epochs 50 --batch_size 256")

    print("Exporting ONNX...")
    run_cmd("cd Project-RIPRA/rippra && CUDA_VISIBLE_DEVICES='' python ml/export_onnx.py --output_dir /kaggle/working/")

    
    with open("/kaggle/working/status.txt", "w") as f:
        f.write("SUCCESS")

except Exception as e:
    with open("/kaggle/working/error.log", "w") as f:
        f.write(traceback.format_exc())
    print("Failed! Exiting with 0 to preserve logs.")
    sys.exit(0)
