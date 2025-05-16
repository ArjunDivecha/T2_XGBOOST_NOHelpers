import h5py
import numpy as np
import sys

def explore_h5_structure(filename):
    """
    Explore the structure of an H5 file in detail, going deeper into the groups.
    """
    print(f"Exploring file: {filename}")
    
    try:
        with h5py.File(filename, 'r') as f:
            print("\nTop level keys:", list(f.keys()))
            
            # Access feature_sets group
            fs_group = f['feature_sets']
            print("\nFeature sets keys (first 5):", list(fs_group.keys())[:5])
            
            # Access first window
            first_window_key = list(fs_group.keys())[0]
            first_window = fs_group[first_window_key]
            print(f"\nFirst window ({first_window_key}) keys:", list(first_window.keys()))
            
            # Access training and validation data
            training = first_window['training']
            validation = first_window['validation']
            print(f"\nTraining keys (first 5):", list(training.keys())[:5])
            
            # Get first factor
            first_factor_key = list(training.keys())[0]
            first_factor = training[first_factor_key]
            
            # Determine if it's a dataset or a group
            if isinstance(first_factor, h5py.Dataset):
                print(f"\nFirst factor ({first_factor_key}) is a dataset:")
                print(f"  - Shape: {first_factor.shape}")
                print(f"  - Type: {first_factor.dtype}")
                print(f"  - First few values: {np.array(first_factor)[:5] if len(first_factor.shape) > 0 and first_factor.shape[0] >= 5 else np.array(first_factor)}")
            else:
                print(f"\nFirst factor ({first_factor_key}) is a group with keys:", list(first_factor.keys()))
                
                # Go deeper if it's a group
                for key in first_factor.keys():
                    sub_item = first_factor[key]
                    if isinstance(sub_item, h5py.Dataset):
                        print(f"  - {key} shape: {sub_item.shape}, type: {sub_item.dtype}")
                        print(f"  - First few values: {np.array(sub_item)[:5] if len(sub_item.shape) > 0 and sub_item.shape[0] >= 5 else np.array(sub_item)}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_h5_structure('output/S8_Feature_Sets.h5')
