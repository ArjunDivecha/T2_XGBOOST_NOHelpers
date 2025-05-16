import h5py
import sys

def print_attrs(name, obj):
    print(f"Path: {name}")
    if isinstance(obj, h5py.Dataset):
        print(f"  Type: Dataset, Shape: {obj.shape}, Dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"  Type: Group, Keys: {list(obj.keys())[:5]}...")

# Open the file
file_path = 'output/S8_Feature_Sets.h5'
print(f"Examining file: {file_path}")

try:
    with h5py.File(file_path, 'r') as f:
        print("\nTop level keys:", list(f.keys()))
        print("\nDetailed structure:")
        f.visititems(print_attrs)
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
