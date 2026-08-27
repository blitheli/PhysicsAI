import torch
import torch.nn as nn
import torch.optim as optim
import time

torch.manual_seed(0)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

device = torch.device("cuda:0")
print("Using:", device)
print("GPU:", torch.cuda.get_device_name(0))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

model = nn.Sequential(
    nn.Linear(16384, 32768),
    nn.ReLU(),
    nn.Linear(32768, 16384),
    nn.ReLU(),
    nn.Linear(16384, 4096),
    nn.ReLU(),
    nn.Linear(4096, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).to(device)

batch_size = 2048
steps = 100
lr = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

start = time.time()
for step in range(1, steps + 1):
    x = torch.randn(batch_size, 16384, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"step={step:03d} loss={loss.item():.4f}")

elapsed = time.time() - start
print(f"elapsed={elapsed:.2f}s")
print("Benchmark finished")