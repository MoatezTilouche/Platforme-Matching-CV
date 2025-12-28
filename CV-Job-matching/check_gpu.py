"""
Script to check GPU/CPU usage in your environment
"""
import sys

print("=" * 60)
print("GPU/CPU Detection Check")
print("=" * 60)

# Check PyTorch
try:
    import torch
    print("\n✓ PyTorch installed")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Device capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("  ⚠ CUDA not available - using CPU")
except ImportError:
    print("\n✗ PyTorch not installed")

# Check TensorFlow
try:
    import tensorflow as tf
    print("\n✓ TensorFlow installed")
    print(f"  Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"    - {gpu}")
    if not gpus:
        print("  ⚠ No GPU detected - using CPU")
except ImportError:
    print("\n✗ TensorFlow not installed")

# Check sentence-transformers (commonly used for embeddings)
try:
    from sentence_transformers import SentenceTransformer
    print("\n✓ sentence-transformers installed")
    # Check what device it would use
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Default device: {device}")
except ImportError:
    print("\n✗ sentence-transformers not installed")

# Check transformers
try:
    import transformers
    print("\n✓ transformers installed")
    print(f"  Version: {transformers.__version__}")
except ImportError:
    print("\n✗ transformers not installed")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"Python version: {sys.version}")

try:
    import torch
    if torch.cuda.is_available():
        print("✓ GPU will be used (CUDA available)")
    else:
        print("✗ CPU will be used (CUDA not available)")
except:
    print("? Cannot determine (PyTorch not installed)")

print("=" * 60)
