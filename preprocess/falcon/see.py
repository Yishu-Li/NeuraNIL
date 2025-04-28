import torch

# Change this to your actual .pt file path if needed
pt_path = 'preprocess/falcon/000950.pt'  # or wherever your file is

# Load the .pt file
data = torch.load(pt_path)

# Check what's inside
print(f"Top-level keys: {list(data.keys())}")

# If there is a 'day' or 'day_labels' field:
if 'day' in data:
    print("Unique days:", torch.unique(data['day']))
elif 'day_labels' in data:
    print("Unique day labels:", torch.unique(data['day_labels']))
else:
    print("No explicit day information found, assuming single day (Day 0)")
