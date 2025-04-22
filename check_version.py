import subprocess

libraries = [
    "httpx",
    "numpy",
    "opencv-python",  # cv2 is installed as opencv-python
    "fastapi",
    "onnxruntime",
    "torch",
    "torchvision",    # Often installed alongside torch for facenet_pytorch
    "facenet-pytorch", # Note the hyphen for pip package name
    "scipy"
]

for lib in libraries:
    try:
        subprocess.run(["pip", "uninstall", lib, "-y"], check=True)
        print(f"Successfully uninstalled {lib}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to uninstall {lib}: {e}")

print("Note: asyncio, time, os, and base64 are part of Python's standard library and cannot be uninstalled.")