"""
Environment Verification Script for DinoTrackerProject
Run this script to verify all required packages are installed and working.
"""

import sys

def check_package(package_name, import_name=None):
    """Try to import a package and report its version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK]  {package_name:20s} - version {version}")
        return True
    except ImportError as e:
        print(f"[MISSING] {package_name:20s} - NOT FOUND")
        return False

print("="*60)
print("DinoTrackerProject Environment Verification")
print("="*60)
print(f"\nPython Version: {sys.version.split()[0]}")
print(f"Python Path: {sys.executable}\n")

print("Checking required packages:\n")

packages = [
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('xformers', 'xformers'),
    ('opencv-python', 'cv2'),
    ('antialiased_cnns', 'antialiased_cnns'),
    ('einops', 'einops'),
    ('imageio', 'imageio'),
    ('kornia', 'kornia'),
    ('matplotlib', 'matplotlib'),
    ('mediapy', 'mediapy'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('pillow', 'PIL'),
    ('tqdm', 'tqdm'),
    ('PyYAML', 'yaml'),
]

all_installed = True
for pkg_name, import_name in packages:
    if not check_package(pkg_name, import_name):
        all_installed = False

print("\n" + "="*60)

# Additional PyTorch checks
try:
    import torch
    print("\nPyTorch Details:")
    print(f"  - Version: {torch.__version__}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - GPU Count: {torch.cuda.device_count()}")
        print(f"  - Current Device: {torch.cuda.current_device()}")
        print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("  - Note: Running on CPU (no GPU support)")
except Exception as e:
    print(f"\n[WARNING] Error checking PyTorch: {e}")

print("\n" + "="*60)

if all_installed:
    print("\n[SUCCESS] All required packages are installed!")
    print("Your environment is ready to use.\n")
else:
    print("\n[FAILURE] Some packages are missing.")
    print("Run: pip install -r requirements.txt\n")

print("="*60)

