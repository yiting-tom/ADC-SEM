import torch
import torch.nn as nn
from typing import Optional


class KANLayer(nn.Module):
    """
    KAN-like layer using piecewise-linear 1D spline basis per input feature
    with learnable coefficients and an optional linear skip connection.

    For each output unit o and input feature i, we parameterize K knot
    coefficients w[o, i, k]. Given input x_i, we compute its position in
    the uniform knot grid and linearly interpolate between the two adjacent
    coefficients (hat basis). We then sum over inputs and add a bias.

    y_o = sum_i ( (1 - t)*w[o,i,idx] + t*w[o,i,idx+1] ) + b_o + (W_skip x)_o

    This approximates the Kolmogorov-Arnold decomposition with a practical,
    efficient implementation for projection heads.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_knots: int = 16,
        x_min: float = -3.0,
        x_max: float = 3.0,
        use_skip: bool = True,
    ):
        super().__init__()
        assert num_knots >= 2, "num_knots must be >= 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_knots = num_knots
        self.register_buffer("x_min", torch.tensor(float(x_min)))
        self.register_buffer("x_max", torch.tensor(float(x_max)))
        self.use_skip = use_skip

        # Spline weights: [out, in, K]
        self.weights = nn.Parameter(
            torch.empty(output_dim, input_dim, num_knots)
        )
        # Optional linear skip connection
        self.skip = nn.Linear(input_dim, output_dim, bias=False) if use_skip else None
        # Bias per output
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize spline weights near zero to start close to linear behavior
        nn.init.trunc_normal_(self.weights, std=0.02)
        if self.skip is not None:
            nn.init.xavier_uniform_(self.skip.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, D_in)

        Returns:
            Tensor of shape (B, D_out)
        """
        assert x.dim() == 2 and x.size(1) == self.input_dim, (
            f"KANLayer expected (B, {self.input_dim}), got {tuple(x.shape)}"
        )

        B, D_in = x.shape
        K = self.num_knots

        # Normalize inputs into [0, 1] based on fixed range, then map to [0, K-1]
        # Clamp to avoid indexing beyond last knot
        t = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        t = t.clamp(0.0, 1.0)
        pos = t * (K - 1)
        idx0 = torch.floor(pos).to(torch.long)
        idx1 = torch.clamp(idx0 + 1, max=K - 1)
        frac = (pos - idx0.to(pos.dtype)).clamp(0.0, 1.0)

        # Prepare for gather: expand weights to (B, out, in, K)
        W = self.weights.unsqueeze(0).expand(B, -1, -1, -1)
        # Expand indices to (B, out, in, 1) for gather along last dim
        idx0_exp = idx0.unsqueeze(1).unsqueeze(-1).expand(B, self.output_dim, D_in, 1)
        idx1_exp = idx1.unsqueeze(1).unsqueeze(-1).expand(B, self.output_dim, D_in, 1)

        W0 = torch.gather(W, dim=3, index=idx0_exp).squeeze(-1)  # (B, out, in)
        W1 = torch.gather(W, dim=3, index=idx1_exp).squeeze(-1)  # (B, out, in)

        # Interpolate weights
        frac_exp = frac.unsqueeze(1)  # (B, 1, in)
        contrib = (1.0 - frac_exp) * W0 + frac_exp * W1  # (B, out, in)

        # Sum over input features
        y_spline = contrib.sum(dim=2)  # (B, out)

        # Add bias and optional linear skip
        y = y_spline + self.bias
        if self.skip is not None:
            y = y + self.skip(x)

        return y


class KANProjectionHead(nn.Module):
    """
    Simple 2-layer KAN projection head for SSL:
      - KANLayer(input_dim -> hidden_dim)
      - Nonlinearity (GELU)
      - KANLayer(hidden_dim -> output_dim)

    Suitable as a drop-in replacement for an MLP projector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: int = 128,
        num_knots: int = 16,
        x_min: float = -3.0,
        x_max: float = 3.0,
        use_skip: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.layer1 = KANLayer(input_dim, hidden_dim, num_knots=num_knots, x_min=x_min, x_max=x_max, use_skip=use_skip)
        self.act = nn.GELU()
        self.layer2 = KANLayer(hidden_dim, output_dim, num_knots=num_knots, x_min=x_min, x_max=x_max, use_skip=use_skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x

