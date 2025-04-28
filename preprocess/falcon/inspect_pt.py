### This was useful to do the preprocessing, it was mostly for Carlos's use 
import torch
from pathlib import Path

pt = Path(__file__).parent / "000950.pt"
data = torch.load(pt)
print("Keys in .pt:", list(data.keys()))
