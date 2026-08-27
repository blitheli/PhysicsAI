import math

import torch
from torch import nn

# 无额外依赖的 1D Poisson 方程求解器

class PINN(nn.Module):
	def __init__(self, hidden_size: int = 64, hidden_layers: int = 3) -> None:
		super().__init__()
		layers: list[nn.Module] = [nn.Linear(1, hidden_size), nn.Tanh()]
		for _ in range(hidden_layers - 1):
			layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
		layers.append(nn.Linear(hidden_size, 1))
		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)


def forcing(x: torch.Tensor) -> torch.Tensor:
	return (math.pi**2) * torch.sin(math.pi * x)


def exact_solution(x: torch.Tensor) -> torch.Tensor:
	return torch.sin(math.pi * x)


def physics_loss(model: PINN, interior_points: torch.Tensor) -> torch.Tensor:
	interior_points.requires_grad_(True)
	u = model(interior_points)
	du_dx = torch.autograd.grad(
		u,
		interior_points,
		grad_outputs=torch.ones_like(u),
		create_graph=True,
	)[0]
	d2u_dx2 = torch.autograd.grad(
		du_dx,
		interior_points,
		grad_outputs=torch.ones_like(du_dx),
		create_graph=True,
	)[0]
	residual = d2u_dx2 + forcing(interior_points)
	return torch.mean(residual.pow(2))


def boundary_loss(model: PINN, device: torch.device) -> torch.Tensor:
	boundary_points = torch.tensor([[0.0], [1.0]], device=device)
	boundary_values = model(boundary_points)
	return torch.mean(boundary_values.pow(2))


def train(model: PINN, device: torch.device, steps: int = 5000) -> None:
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	for step in range(1, steps + 1):
		interior_points = torch.rand(256, 1, device=device)
		loss_pde = physics_loss(model, interior_points)
		loss_bc = boundary_loss(model, device)
		loss = loss_pde + 10.0 * loss_bc

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 500 == 0:
			print(
				f"step={step:5d} total={loss.item():.6f} "
				f"pde={loss_pde.item():.6f} bc={loss_bc.item():.6f}"
			)


def evaluate(model: PINN, device: torch.device) -> None:
	grid = torch.linspace(0.0, 1.0, 200, device=device).unsqueeze(-1)
	prediction = model(grid)
	target = exact_solution(grid)
	max_error = torch.max(torch.abs(prediction - target)).item()
	mse = torch.mean((prediction - target).pow(2)).item()
	print(f"evaluation mse={mse:.6e} max_error={max_error:.6e}")


def main() -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = PINN().to(device)
	train(model, device)
	evaluate(model, device)


if __name__ == "__main__":
	main()
