import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parent / "physicsnemo"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import physicsnemo
from physicsnemo.models.mlp.fully_connected import FullyConnected

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullyConnected(in_features=32, out_features=64).to(device)
input_tensor = torch.randn(128, 32, device=device)
output = model(input_tensor)
print("模型输出形状:", output.shape)
