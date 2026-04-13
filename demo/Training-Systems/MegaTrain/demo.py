from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


torch.manual_seed(0)


# ------------------------------------------------------------
# MegaTrain backbone demo
# ------------------------------------------------------------
# Major system / algorithm pieces implemented:
#   1) host-resident parameters and optimizer state
#   2) stateless layer templates
#   3) streamed per-layer weight binding to device
#   4) simplified double-buffered prefetch structure
#   5) training step with gradients returned to host
#
# This is a simplified implementation of the core MegaTrain idea:
# persistent state lives in host memory, while the GPU acts as a transient
# compute engine for streamed layers.
# ------------------------------------------------------------


@dataclass
class HostLayerState:
    weight: torch.Tensor      # persistent on CPU
    bias: torch.Tensor        # persistent on CPU
    mom_w: torch.Tensor       # optimizer state on CPU
    mom_b: torch.Tensor       # optimizer state on CPU


def init_host_layer(in_dim: int, out_dim: int) -> HostLayerState:
    w = torch.randn(out_dim, in_dim, device="cpu") * 0.02
    b = torch.zeros(out_dim, device="cpu")
    mom_w = torch.zeros_like(w, device="cpu")
    mom_b = torch.zeros_like(b, device="cpu")
    return HostLayerState(weight=w, bias=b, mom_w=mom_w, mom_b=mom_b)


def stateless_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Stateless layer template:
    computation is fixed, weights are bound dynamically when streamed in.
    """
    return F.linear(x, weight, bias)


class MegaTrainBackbone:
    def __init__(self, dims: List[int], device: torch.device):
        self.device = device
        self.layers: List[HostLayerState] = []
        for i in range(len(dims) - 1):
            self.layers.append(init_host_layer(dims[i], dims[i + 1]))

    def _stream_to_device(
        self,
        layer: HostLayerState,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stream host-resident parameters to device.
        """
        w = layer.weight.to(self.device, non_blocking=True).detach(
        ).requires_grad_(requires_grad)
        b = layer.bias.to(self.device, non_blocking=True).detach(
        ).requires_grad_(requires_grad)
        return w, b

    def forward_streamed(self, x: torch.Tensor):
        """
        Simplified streamed forward with a double-buffer-like structure:
        prefetch next layer while computing current layer.

        We keep this concise by expressing the buffering pattern in code structure
        rather than building a full multi-stream runtime.
        """
        bound_params = []

        # prefetch first layer
        current_w, current_b = self._stream_to_device(self.layers[0])

        h = x
        for i in range(len(self.layers)):
            # prefetch next layer early if it exists
            next_w, next_b = (None, None)
            if i + 1 < len(self.layers):
                next_w, next_b = self._stream_to_device(self.layers[i + 1])

            # compute current layer
            h = stateless_linear(h, current_w, current_b)
            if i + 1 < len(self.layers):
                h = F.gelu(h)

            bound_params.append((current_w, current_b))

            # advance buffer
            current_w, current_b = next_w, next_b

        return h, bound_params

    def apply_host_update(
        self,
        bound_params: List[Tuple[torch.Tensor, torch.Tensor]],
        lr: float = 1e-2,
        momentum: float = 0.9,
    ) -> None:
        """
        Pull grads back to host and update persistent host weights + optimizer state.
        """
        for layer_state, (w_dev, b_dev) in zip(self.layers, bound_params):
            grad_w = w_dev.grad.detach().cpu()
            grad_b = b_dev.grad.detach().cpu()

            layer_state.mom_w = momentum * layer_state.mom_w + grad_w
            layer_state.mom_b = momentum * layer_state.mom_b + grad_b

            layer_state.weight -= lr * layer_state.mom_w
            layer_state.bias -= lr * layer_state.mom_b


def make_batch(batch_size: int = 32, in_dim: int = 64, num_classes: int = 10, device: torch.device | None = None):
    x = torch.randn(batch_size, in_dim, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def train_step(model: MegaTrainBackbone, x: torch.Tensor, y: torch.Tensor, lr: float = 1e-2):
    logits, bound = model.forward_streamed(x)
    loss = F.cross_entropy(logits, y)

    # grads only exist for streamed device tensors, not persistent host master weights
    loss.backward()

    model.apply_host_update(bound_params=bound, lr=lr)
    return float(loss.item())


def inspect_host_memory(model: MegaTrainBackbone) -> None:
    total_param_bytes = 0
    total_opt_bytes = 0

    for layer in model.layers:
        total_param_bytes += layer.weight.numel() * layer.weight.element_size()
        total_param_bytes += layer.bias.numel() * layer.bias.element_size()

        total_opt_bytes += layer.mom_w.numel() * layer.mom_w.element_size()
        total_opt_bytes += layer.mom_b.numel() * layer.mom_b.element_size()

    print(f"host param bytes     : {total_param_bytes / 1024:.2f} KB")
    print(f"host optimizer bytes : {total_opt_bytes / 1024:.2f} KB")
    print(
        f"persistent host total: {(total_param_bytes + total_opt_bytes) / 1024:.2f} KB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # modest dimensions for a runnable backbone
    dims = [64, 128, 256, 128, 10]
    model = MegaTrainBackbone(dims=dims, device=device)

    print("=== MegaTrain backbone demo ===\n")
    print(f"device: {device}")
    inspect_host_memory(model)
    print()

    for step in range(5):
        x, y = make_batch(batch_size=32, in_dim=64,
                          num_classes=10, device=device)
        loss = train_step(model, x, y, lr=5e-2)
        print(f"step={step:02d} loss={loss:.4f}")

    print("\nInterpretation:")
    print("- Parameters and optimizer state persist on host memory.")
    print("- Each layer's weights are streamed to device only when needed.")
    print("- Stateless layer templates bind streamed weights dynamically.")
    print("- Gradients are returned to host, where the master weights are updated.")
    print("- This is the core backbone of MegaTrain's memory-centric training design.")


if __name__ == "__main__":
    main()
