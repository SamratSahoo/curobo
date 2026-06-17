#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Third Party
import torch

# CuRobo
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .cost_base import CostBase


@get_torch_jit_decorator()
def uniform_velocity_cost(vels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # vels: [batch, horizon, dof]
    # per-timestep squared joint-space speed (smooth everywhere, no sqrt singularity at v=0)
    speed_sq = torch.sum(torch.square(vels), dim=-1)  # [batch, horizon]
    # mean speed across the horizon (per batch element), broadcast back over horizon
    mean_speed_sq = torch.mean(speed_sq, dim=1, keepdim=True)  # [batch, 1]
    # penalize squared deviation of each timestep's speed from the horizon mean
    cost = weight * torch.square(speed_sq - mean_speed_sq)  # [batch, horizon]
    return cost


@get_torch_jit_decorator()
def uniform_velocity_run_cost(
    vels: torch.Tensor, weight: torch.Tensor, run_weight: torch.Tensor
) -> torch.Tensor:
    # vels: [batch, horizon, dof]; run_weight: [1, horizon]
    speed_sq = torch.sum(torch.square(vels), dim=-1)  # [batch, horizon]
    mean_speed_sq = torch.mean(speed_sq, dim=1, keepdim=True)  # [batch, 1]
    cost = weight * run_weight * torch.square(speed_sq - mean_speed_sq)  # [batch, horizon]
    return cost


class UniformVelocityCost(CostBase):
    """Penalize deviation of per-timestep joint-space speed from its horizon mean.

    Encourages the arm to move at as close to a uniform (constant) joint-space
    velocity as possible across the trajectory. Uses squared joint-space speed
    e_t = sum_dof(v_t^2) as a numerically safe per-timestep measure (smooth
    everywhere; no sqrt singularity at v=0). The cost at timestep t is
    weight * (e_t - mean_t(e))^2. Because the trajectory starts/ends at rest,
    terminal/run_weight (inherited from CostBase) can down-weight the endpoints
    via yaml.
    """

    def forward(self, vels):
        # vels: [batch, horizon, dof]
        if not self.terminal or self.run_weight is None:
            cost = uniform_velocity_cost(vels, self.weight)
        else:
            if self._run_weight_vec is None or self._run_weight_vec.shape[1] != vels.shape[1]:
                self._run_weight_vec = torch.ones(
                    (1, vels.shape[1]),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
                self._run_weight_vec[:, 1:-1] *= self.run_weight
            cost = uniform_velocity_run_cost(vels, self.weight, self._run_weight_vec)
        return cost
