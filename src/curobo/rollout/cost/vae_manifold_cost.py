#
# VAE motion-manifold cost (tamp-vla / tiptop-viz manifold study).
#
# Scores how DROID-like a trajopt segment's joint-space motion is, using the Filterbank VAE
# trained in tamp-vla/vae/ (see vae/train.py, class FilterbankVAE). The segment positions
# [batch, horizon, dof] are resampled from the trajopt rate (base_dt) to the VAE's 15 Hz, the
# joint metric [q|v|a|j] (28-D for 7 joints) is rebuilt by central differencing, standardized
# with the VAE's per-channel stats, and encoded by the filterbank into one latent per segment
# (variable length, masked global pooling -- no windowing).
#
# The score is the squared Mahalanobis distance of that latent to the DROID human-teleop
# latent cluster. MINIMIZING it pulls the segment's motion style toward DROID. (The filterbank
# latent SEPARATES DROID from cuTAMP, so cuTAMP segments start far from the DROID mean.)
#
# Everything -- encoder weights, channel stats, and DROID latent mean/precision -- is loaded
# from one self-contained checkpoint: default = tamp-vla/vae/checkpoints/vae.pt, resolved
# relative to this file (override with the VAE_MANIFOLD_CKPT env var). The DROID stats are
# baked in by vae/train.py::droid_latent_stats.
#
# Third Party
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CuRobo
from curobo.types.base import TensorDeviceType

# Local Folder
from .cost_base import CostBase, CostConfig

# Default checkpoint: tamp-vla/vae/checkpoints/vae.pt, resolved RELATIVE to this file so it
# works on any clone of the tamp-vla monorepo (curobo is a submodule at tamp-vla/curobo).
# .../tamp-vla/curobo/src/curobo/rollout/cost/vae_manifold_cost.py -> parents[5] == tamp-vla.
_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_VAE_MANIFOLD_CKPT = os.environ.get(
    "VAE_MANIFOLD_CKPT", str(_REPO_ROOT / "vae" / "checkpoints" / "vae.pt")
)

# The rate the VAE was trained at (vae/data.py COMMON_RATE); segments are resampled to it.
VAE_RATE_HZ = 15.0


# --------------------------------------------------------------------------- #
# Filterbank VAE replica (mirrors tamp-vla/vae/train.py:FilterbankVAE so the    #
# saved state_dict loads strictly).                                            #
# --------------------------------------------------------------------------- #
def _masked_mean_std(x, m):                                # x:(B,C,T) m:(B,1,T) -> (mean, std)
    s = m.sum(-1).clamp(min=1.0)
    mean = (x * m).sum(-1) / s
    std = (((x - mean.unsqueeze(-1)) ** 2 * m).sum(-1) / s).clamp(min=1e-8).sqrt()
    return mean, std


def _masked_stats(x, m):                                   # masked mean | std | max over time
    mean, std = _masked_mean_std(x, m)
    return torch.cat([mean, std, (x + (1 - m) * -1e9).amax(-1)], 1)


class _FilterbankVAE(nn.Module):
    def __init__(self, ch, d, n_target, emb=96, p=0.2):
        super().__init__()
        specs = [(3, 1), (7, 1), (15, 1), (7, 2), (15, 4), (15, 8)]
        self.fb = nn.ModuleList([nn.Conv1d(ch, 32, k, dilation=dl, padding=dl * (k - 1) // 2)
                                 for k, dl in specs])
        self.fbn = nn.ModuleList([nn.BatchNorm1d(32) for _ in specs])
        self.t = nn.Sequential(
            nn.Conv1d(ch, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, 2, 2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.GELU())
        self.fc = nn.Sequential(nn.Linear(len(specs) * 32 * 2 + 96 * 3, 256), nn.LayerNorm(256),
                                nn.GELU(), nn.Dropout(p),
                                nn.Linear(256, emb), nn.LayerNorm(emb), nn.GELU())
        self.to_lat = nn.Linear(emb, 2 * d)
        self.aux = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, n_target))
        self.d, self.ch = d, ch

    def _embed(self, x, m):
        fbp = []
        for b, bn in zip(self.fb, self.fbn):
            mean, std = _masked_mean_std(bn(b(x)).abs(), m)
            fbp += [mean, std]
        tt = self.t(x)
        mt = m[:, :, ::8][:, :, :tt.shape[-1]]
        if mt.shape[-1] < tt.shape[-1]:
            mt = F.pad(mt, (0, tt.shape[-1] - mt.shape[-1]))
        return self.fc(torch.cat(fbp + [_masked_stats(tt, mt)], 1))

    def encode_mu(self, x, m):
        return self.to_lat(self._embed(x, m)).chunk(2, dim=1)[0]   # latent mean only


# --------------------------------------------------------------------------- #
# checkpoint loading (cached per (path, device, dtype))                        #
# --------------------------------------------------------------------------- #
_PACK_CACHE = {}


def load_vae_manifold(checkpoint_path: str, tensor_args: TensorDeviceType):
    """Load (and cache) the filterbank encoder + channel stats + DROID latent mean/precision."""
    key = (checkpoint_path, str(tensor_args.device), str(tensor_args.dtype))
    if key in _PACK_CACHE:
        return _PACK_CACHE[key]

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "droid_latent_mean" not in blob:
        raise KeyError(
            f"{checkpoint_path} has no DROID latent stats; rebuild vae.pt with "
            "vae/train.py (it bakes them in via droid_latent_stats)."
        )
    ch, d, n_feat = int(blob["ch"]), int(blob["latent"]), int(blob["n_feat"])
    model = _FilterbankVAE(ch, d, n_feat)
    model.load_state_dict(blob["state_dict"])
    model = model.to(device=tensor_args.device, dtype=tensor_args.dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def _t(x, shape=None):
        t = torch.as_tensor(np.asarray(x), device=tensor_args.device, dtype=tensor_args.dtype)
        return t.view(*shape) if shape is not None else t

    pack = {
        "model": model,
        "n_joints": int(blob.get("n_joints", 7)),
        "chan_mu": _t(blob["chan_mu"], (1, 1, ch)),
        "chan_sd": _t(blob["chan_sd"], (1, 1, ch)),
        "droid_mean": _t(blob["droid_latent_mean"], (1, d)),
        "droid_prec": _t(blob["droid_latent_precision"], (d, d)),
    }
    _PACK_CACHE[key] = pack
    return pack


# --------------------------------------------------------------------------- #
# motion -> standardized [q|v|a|j] segment (matches vae/data.py preprocessing)  #
# --------------------------------------------------------------------------- #
def _grad_time(x: torch.Tensor, h: float) -> torch.Tensor:
    """d x / d t along dim=1, central in the interior + one-sided edges -- identical to
    numpy.gradient(edge_order=1), which vae/data.py uses. x: [B, T, C]."""
    interior = (x[:, 2:] - x[:, :-2]) / (2.0 * h)
    first = (x[:, 1:2] - x[:, 0:1]) / h
    last = (x[:, -1:] - x[:, -2:-1]) / h
    return torch.cat([first, interior, last], dim=1)


def positions_to_input(position: torch.Tensor, source_dt: float, pack: dict):
    """[B, H, >=n_joints] joint positions -> ([B, 28, n2] standardized series, [B, 1, n2] mask).

    Resamples the segment from its native rate (1/source_dt) to VAE_RATE_HZ, rebuilds the
    joint metric [q|v|a|j] (28-D for 7 joints) by central differencing, and standardizes with
    the VAE's per-channel stats. The filterbank pools over the whole (variable-length) segment,
    so the mask is all-ones."""
    B, H, _ = position.shape
    J = pack["n_joints"]
    if position.shape[-1] < J:
        raise ValueError(
            f"VAE-manifold cost expects >= {J} joints (the VAE was trained on {J}-DOF Franka "
            f"joint metrics) but got dof={position.shape[-1]}"
        )
    q = position[..., :J]
    n2 = max(2, int(round((H - 1) * float(source_dt) * VAE_RATE_HZ)) + 1)
    if n2 != H:
        q = F.interpolate(q.transpose(1, 2), size=n2, mode="linear", align_corners=True).transpose(1, 2)
    h = 1.0 / VAE_RATE_HZ
    v = _grad_time(q, h)
    a = _grad_time(v, h)
    j = _grad_time(a, h)
    feats = torch.cat([q, v, a, j], dim=-1)                 # [B, n2, 28]
    feats = (feats - pack["chan_mu"]) / pack["chan_sd"]
    x = feats.transpose(1, 2).contiguous()                  # [B, 28, n2]
    mask = torch.ones(B, 1, n2, device=x.device, dtype=x.dtype)
    return x, mask


def segment_maha2(position: torch.Tensor, source_dt: float, pack: dict) -> torch.Tensor:
    """Squared Mahalanobis distance of each segment's latent to the DROID cluster -> [B]."""
    x, mask = positions_to_input(position, source_dt, pack)
    mu = pack["model"].encode_mu(x, mask)
    dz = mu - pack["droid_mean"]
    return torch.einsum("ni,ij,nj->n", dz, pack["droid_prec"], dz)


# --------------------------------------------------------------------------- #
# cost config + module                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class VaeManifoldCostConfig(CostConfig):
    checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT
    n_joints: int = 7
    source_dt: float = 0.15         # trajopt base_dt (gradient_trajopt.yml model.dt_traj_params.base_dt)

    def __post_init__(self):
        return super().__post_init__()


class VaeManifoldCost(CostBase, VaeManifoldCostConfig):
    """Per-segment VAE motion-manifold cost (DROID Mahalanobis distance).

    forward(position) -> [batch, horizon]. Each segment is encoded to one latent; its squared
    Mahalanobis distance to the DROID cluster is spread uniformly over the horizon (so the
    horizon sum equals weight * distance, and gradients flow through the encoder to every
    waypoint). weight == 0 disables the cost (CostBase)."""

    def __init__(self, config: Optional[VaeManifoldCostConfig] = None):
        if config is not None:
            VaeManifoldCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._init_post_config()
        self._pack = None  # lazy-loaded on first enabled forward

    def _get_pack(self):
        if self._pack is None:
            self._pack = load_vae_manifold(self.checkpoint_path, self.tensor_args)
            if self.n_joints != self._pack["n_joints"]:
                self._pack = dict(self._pack, n_joints=self.n_joints)
        return self._pack

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        # position: [batch, horizon, dof]
        horizon = position.shape[1]
        pack = self._get_pack()
        # curobo's clique tensor-step C++ backward asserts grad w.r.t. position is contiguous, but our
        # resample -> transpose chain hands back a NON-contiguous grad; coerce it via a hook.
        if position.requires_grad:
            position.register_hook(lambda g: g.contiguous() if g is not None else g)
        seg = segment_maha2(position, self.source_dt, pack)  # [batch]
        # repeat (not expand) -> contiguous cost tensor for the downstream cat_sum / reductions
        return (self.weight * (seg / horizon)).unsqueeze(1).repeat(1, horizon)


# --------------------------------------------------------------------------- #
# per-segment trace (for the tiptop-viz cost-over-time plot)                   #
# --------------------------------------------------------------------------- #
def trajectory_score_trace(
    position: torch.Tensor,
    source_dt: float,
    *,
    checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT,
    n_joints: int = 7,
    tensor_args: Optional[TensorDeviceType] = None,
):
    """Per-timestep VAE-manifold score for one trajectory segment (length == T).

    The filterbank scores the WHOLE segment as one scalar (no per-window decomposition), so
    the trace is that scalar broadcast over the segment's T timesteps -- exactly the cost the
    optimizer sees. position: [T, dof]. Returns a float64 numpy array [T]."""
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    pack = dict(load_vae_manifold(checkpoint_path, tensor_args), n_joints=n_joints)
    if isinstance(position, torch.Tensor):
        pos = position.detach().to(device=tensor_args.device, dtype=tensor_args.dtype)
    else:
        pos = torch.as_tensor(np.asarray(position), device=tensor_args.device, dtype=tensor_args.dtype)
    T = pos.shape[0]
    with torch.no_grad():
        s = float(segment_maha2(pos.unsqueeze(0), source_dt, pack).item())
    return np.full(T, s, dtype=np.float64)
