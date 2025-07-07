from scipy.io import loadmat

# Load a sample .mat file
mat_data = loadmat('data/eeg_record1.mat')

# Print all top-level keys in the file
print("Keys in .mat file:")
print(mat_data.keys())

# Show one key’s contents (pick any real one, not '__header__', etc.)
for key in mat_data:
    if not key.startswith('__'):
        print(f"\nPreview of '{key}':")
        print(mat_data[key])
