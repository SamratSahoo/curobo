#
# VAE motion-manifold cost (added for the tamp-vla / tiptop-viz manifold study).
#
# Scores how a trajopt segment's joint-space motion relates to the DROID human-teleop
# motion manifold, using a beta-VAE trained on DROID joint metrics (see ../../../../../analysis/,
# train_droid_vae_all.py [position-aware q+v+a+j, default] / train_droid_vae_nocollapse.py [v+a+j]).
#
# The encoder + channel statistics + DROID latent statistics are loaded from a single
# self-contained artifact built by analysis (default DEFAULT_VAE_MANIFOLD_CKPT). To stay
# faithful to the VAE's training distribution, the trajopt segment (horizon=32 @ dt=0.15s,
# i.e. ~6.67 Hz) is resampled to the VAE's 15 Hz, then the joint metric [q|v|a|j] is rebuilt
# with central differences (matching numpy.gradient used in analysis), the VAE's channel_slice
# is selected (0:28=[q,v,a,j] for the "all" VAE, 7:28=[v,a,j] for the vaj VAE), standardized
# with the DROID per-channel stats, and cut into the VAE's 30-step windows.
#
# IMPORTANT (measured, see PR notes): DROID sits at HIGH KL/recon/Mahalanobis; cuTAMP
# trajectories are smooth and already sit LOW (near the latent centre, well reconstructed).
# So the "membership" modes (kl/recon/maha2), when MINIMIZED, act as stay-in-distribution
# regularizers (they bound drift into implausible motion) -- they do NOT pull toward DROID,
# because cuTAMP already minimizes them. The "toward-DROID" modes (kl_hinge/maha2_shell)
# push the encoding OUTWARD toward DROID's characteristic level instead.
#
# Third Party
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CuRobo
from curobo.types.base import TensorDeviceType

# Local Folder
from .cost_base import CostBase, CostConfig

# Default artifact (encoder weights + chan stats + DROID latent stats), built by
# analysis/ (see /tmp build script in PR notes). Overridable via the cost config.
DEFAULT_VAE_MANIFOLD_CKPT = os.environ.get(
    "VAE_MANIFOLD_CKPT",
    # position-aware "all" VAE (joint q+v+a+j, 28 ch, channel_slice 0:28). The pure-vaj artifact
    # (betavae_droid_manifold_cost.pt, channel_slice 7:28) also loads via the channel_slice path.
    "/home/samrat/tamp-vla/tamp-vla/analysis/outputs/cache/betavae_droid_manifold_cost_all.pt",
)

# The rate everything in analysis is resampled to before differencing (config.COMMON_RATE).
VAE_RATE_HZ = 15.0

VALID_MODES = ("kl", "recon", "maha2", "kl_hinge", "maha2_shell")


# --------------------------------------------------------------------------- #
# beta-VAE replica (architecture mirrors analysis/learned.py: ConvEnc + VAE)   #
# --------------------------------------------------------------------------- #
class _ConvEnc(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class _VAE(nn.Module):
    """Same module as analysis/learned.py:VAE so the saved state_dict loads strictly."""

    def __init__(self, ch: int, d: int, win: int):
        super().__init__()
        self.enc = _ConvEnc(ch)
        self.fc = nn.Linear(64, 2 * d)
        self.dec_fc = nn.Linear(d, 64 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1), nn.GELU(),
            nn.ConvTranspose1d(32, ch, 5, stride=2, padding=2, output_padding=1),
        )
        self.d = d
        self.win = win

    def encode(self, x):
        mu, logvar = self.fc(self.enc(x)).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 64, 4)
        return self.dec(h)[..., : self.win]


# --------------------------------------------------------------------------- #
# artifact loading (cached per (path, device, dtype))                          #
# --------------------------------------------------------------------------- #
_PACK_CACHE = {}


def load_vae_manifold(checkpoint_path: str, tensor_args: TensorDeviceType):
    """Load (and cache) the VAE encoder + statistics needed to score motion windows."""
    key = (checkpoint_path, str(tensor_args.device), str(tensor_args.dtype))
    if key in _PACK_CACHE:
        return _PACK_CACHE[key]

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ch, d, win = int(blob["ch"]), int(blob["latent"]), int(blob["window"])
    model = _VAE(ch, d, win)
    model.load_state_dict(blob["state_dict"])
    model = model.to(device=tensor_args.device, dtype=tensor_args.dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def _t(x, shape=None):
        t = torch.as_tensor(np.asarray(x), device=tensor_args.device, dtype=tensor_args.dtype)
        return t.view(*shape) if shape is not None else t

    pack = {
        "model": model,
        "ch": ch,
        "d": d,
        "window": win,
        "stride": int(blob["stride"]),
        "n_joints": int(blob.get("n_joints", 7)),
        # which channels of the joint metric [q|v|a|j] (0:28) the VAE consumes:
        # [7,28]=[v,a,j] (vaj VAE) or [0,28]=[q,v,a,j] (position-aware "all" VAE).
        "channel_slice": list(blob.get("channel_slice", [7, 28])),
        "chan_mu": _t(blob["chan_mu"], (1, 1, ch)),
        "chan_sd": _t(blob["chan_sd"], (1, 1, ch)),
        "droid_mean": _t(blob["droid_latent_mean"], (1, d)),
        "droid_prec": _t(blob["droid_latent_precision"], (d, d)),
        "kl_target": float(blob.get("kl_droid_median", 6.4)),
    }
    _PACK_CACHE[key] = pack
    return pack


# --------------------------------------------------------------------------- #
# motion -> standardized 30-step windows (matches analysis preprocessing)      #
# --------------------------------------------------------------------------- #
def _grad_time(x: torch.Tensor, h: float) -> torch.Tensor:
    """d x / d t along dim=1. Central in the interior, one-sided at the edges --
    identical to numpy.gradient(edge_order=1), which analysis uses. x: [B, T, C]."""
    interior = (x[:, 2:] - x[:, :-2]) / (2.0 * h)
    first = (x[:, 1:2] - x[:, 0:1]) / h
    last = (x[:, -1:] - x[:, -2:-1]) / h
    return torch.cat([first, interior, last], dim=1)


def positions_to_windows(position: torch.Tensor, source_dt: float, pack: dict):
    """[B, H, >=n_joints] joint positions -> standardized VAE windows.

    Resamples the segment from its native rate (1/source_dt) to VAE_RATE_HZ, recomputes
    velocity/accel/jerk by central differencing, builds the joint metric [q|v|a|j] (28-D for
    7 joints), selects the VAE's channel_slice (7:28 = [v,a,j] for the vaj VAE, or 0:28 =
    [q,v,a,j] for the position-aware "all" VAE), standardizes with the DROID channel stats,
    and cuts overlapping (window, stride) windows that fully cover the segment.

    Returns (windows [B*nw, ch, window], B, nw, starts, n2) where starts/n2 are in the
    resampled (15 Hz) index space (used by the per-timestep trace helper).
    """
    B, H, _ = position.shape
    J = pack["n_joints"]
    win, stride, ch = pack["window"], pack["stride"], pack["ch"]
    cs0, cs1 = pack["channel_slice"]
    if position.shape[-1] < J:
        raise ValueError(
            f"VAE-manifold cost expects >= {J} joints (the VAE was trained on {J}-DOF Franka "
            f"joint metrics) but got dof={position.shape[-1]}"
        )
    q = position[..., :J]  # [B, H, J]

    # resample positions to 15 Hz (differentiable, linear)
    n2 = max(2, int(round((H - 1) * float(source_dt) * VAE_RATE_HZ)) + 1)
    if n2 != H:
        q = F.interpolate(q.transpose(1, 2), size=n2, mode="linear", align_corners=True).transpose(1, 2)
    new_h = 1.0 / VAE_RATE_HZ

    v = _grad_time(q, new_h)
    a = _grad_time(v, new_h)
    j = _grad_time(a, new_h)
    # joint metric [q|v|a|j] (channels 0:28 of analysis' metric series), then the VAE's slice.
    feats = torch.cat([q, v, a, j], dim=-1)[..., cs0:cs1]  # [B, n2, ch]

    # window first (zero-pad a too-short segment in RAW space, like analysis), then standardize.
    if n2 < win:
        feats = F.pad(feats, (0, 0, 0, win - n2))  # pad time dim at the end with zeros
        n2 = win
        starts = [0]
    else:
        starts = list(range(0, n2 - win + 1, stride))
        if starts[-1] != n2 - win:
            starts.append(n2 - win)

    wins = torch.stack([feats[:, s : s + win, :] for s in starts], dim=1)  # [B, nw, win, ch]
    nw = wins.shape[1]
    wins = (wins - pack["chan_mu"].unsqueeze(1)) / pack["chan_sd"].unsqueeze(1)
    wins = wins.reshape(B * nw, win, ch).transpose(1, 2).contiguous()  # [B*nw, ch, win]
    return wins, B, nw, starts, n2


def score_from_latents(mu, logvar, wins, pack, mode, target):
    """Per-window scalar score [N] from already-encoded latents. Lower is better (minimized)."""
    if mode == "kl":
        # KL(q(z|x) || N(0,I)) -- minimizing pulls the encoding toward the prior centre.
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)
    if mode == "kl_hinge":
        # push the encoding ENERGY up toward DROID's typical KL (toward DROID).
        tgt = pack["kl_target"] if target is None else float(target)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)
        return F.relu(tgt - kl).pow(2)
    if mode == "recon":
        rec = pack["model"].decode(mu)
        return (rec - wins).pow(2).mean(dim=(1, 2))
    if mode == "maha2":
        dz = mu - pack["droid_mean"]
        return torch.einsum("ni,ij,nj->n", dz, pack["droid_prec"], dz)
    if mode == "maha2_shell":
        # push the latent OUT to the DROID covariance shell (E[maha2] = d for a DROID sample).
        tgt = float(pack["d"]) if target is None else float(target)
        dz = mu - pack["droid_mean"]
        m2 = torch.einsum("ni,ij,nj->n", dz, pack["droid_prec"], dz)
        return F.relu(tgt - m2).pow(2)
    raise ValueError(f"unknown vae_manifold mode {mode!r}; expected one of {VALID_MODES}")


def window_scores(wins: torch.Tensor, pack: dict, mode: str, target: Optional[float]) -> torch.Tensor:
    """Per-window scalar score [N] (encodes then scores). Lower is better for ALL modes."""
    mu, logvar = pack["model"].encode(wins)  # [N, d], [N, d]
    return score_from_latents(mu, logvar, wins, pack, mode, target)


# --------------------------------------------------------------------------- #
# cost config + module                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class VaeManifoldCostConfig(CostConfig):
    checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT
    mode: str = "kl_hinge"
    target: Optional[float] = None  # None -> mode default (kl_hinge: DROID median KL; maha2_shell: d)
    n_joints: int = 7
    source_dt: float = 0.15  # trajopt base_dt (gradient_trajopt.yml model.dt_traj_params.base_dt)

    def __post_init__(self):
        return super().__post_init__()


class VaeManifoldCost(CostBase, VaeManifoldCostConfig):
    """Per-segment VAE motion-manifold cost.

    forward(position) -> [batch, horizon]. The per-segment score is the mean over the
    segment's VAE windows; it is spread uniformly over the horizon so that the horizon
    sum equals weight * mean_window_score (gradients flow through the encoder to every
    waypoint). weight == 0 disables the cost (CostBase).
    """

    def __init__(self, config: Optional[VaeManifoldCostConfig] = None):
        if config is not None:
            VaeManifoldCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._init_post_config()
        if self.mode not in VALID_MODES:
            raise ValueError(f"vae_manifold mode {self.mode!r} not in {VALID_MODES}")
        self._pack = None  # lazy-loaded on first enabled forward

    def _get_pack(self):
        if self._pack is None:
            self._pack = load_vae_manifold(self.checkpoint_path, self.tensor_args)
            if self.n_joints != self._pack["n_joints"]:
                # config wins, but warn-by-overwrite: the artifact was built for n_joints.
                self._pack = dict(self._pack, n_joints=self.n_joints)
        return self._pack

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        # position: [batch, horizon, dof]
        batch, horizon = position.shape[0], position.shape[1]
        pack = self._get_pack()
        # curobo's clique tensor-step C++ backward asserts grad w.r.t. position is contiguous, but our
        # resample -> transpose -> (full-width) slice chain hands back a NON-contiguous grad, which
        # crashes backward_step_position_clique2. Coerce the accumulated grad to contiguous via a hook.
        if position.requires_grad:
            position.register_hook(lambda g: g.contiguous() if g is not None else g)
        wins, B, nw, _, _ = positions_to_windows(position, self.source_dt, pack)
        scores = window_scores(wins, pack, self.mode, self.target)  # [B*nw]
        seg = scores.view(B, nw).mean(dim=1)  # [batch]
        # repeat (not expand) -> contiguous cost tensor for the downstream cat_sum / reductions
        cost = (self.weight * (seg / horizon)).unsqueeze(1).repeat(1, horizon)
        return cost


# --------------------------------------------------------------------------- #
# per-timestep trace (for the tiptop-viz cost-over-time plot)                  #
# --------------------------------------------------------------------------- #
def _prep_segment(position, source_dt, checkpoint_path, n_joints, tensor_args):
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    pack = dict(load_vae_manifold(checkpoint_path, tensor_args), n_joints=n_joints)
    if isinstance(position, torch.Tensor):
        pos = position.detach().to(device=tensor_args.device, dtype=tensor_args.dtype)
    else:
        pos = torch.as_tensor(np.asarray(position), device=tensor_args.device, dtype=tensor_args.dtype)
    return pack, pos, tensor_args


def _scatter_interp(scores, starts, n2, win, T, tensor_args):
    """Overlap-average per-window scores onto the 15 Hz grid, then resample to T timesteps."""
    trace15 = torch.zeros(n2, device=tensor_args.device, dtype=tensor_args.dtype)
    count = torch.zeros(n2, device=tensor_args.device, dtype=tensor_args.dtype)
    for wi, s in enumerate(starts):
        trace15[s : s + win] += scores[wi]
        count[s : s + win] += 1.0
    trace15 = trace15 / count.clamp(min=1.0)
    traceT = F.interpolate(trace15.view(1, 1, -1), size=T, mode="linear", align_corners=True).view(-1)
    return traceT.double().cpu().numpy()


def trajectory_score_trace(
    position: torch.Tensor,
    source_dt: float,
    *,
    checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT,
    mode: str = "kl_hinge",
    target: Optional[float] = None,
    n_joints: int = 7,
    tensor_args: Optional[TensorDeviceType] = None,
):
    """Per-timestep VAE-manifold score for one trajectory segment + mode, length == position's T.

    position: [T, dof]. Windows are scored (weight = 1), overlap-averaged onto the 15 Hz
    sample grid, then resampled back to the segment's T timesteps so it aligns with the
    plan's positions/velocities on the plot. Returns a float64 numpy array [T].
    """
    pack, pos, tensor_args = _prep_segment(position, source_dt, checkpoint_path, n_joints, tensor_args)
    T = pos.shape[0]
    with torch.no_grad():
        wins, _, nw, starts, n2 = positions_to_windows(pos.unsqueeze(0), source_dt, pack)
        scores = window_scores(wins, pack, mode, target)  # [nw]
        return _scatter_interp(scores, starts, n2, pack["window"], T, tensor_args)


def trajectory_score_traces(
    position: torch.Tensor,
    source_dt: float,
    *,
    checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT,
    modes=VALID_MODES,
    target_overrides: Optional[dict] = None,
    n_joints: int = 7,
    tensor_args: Optional[TensorDeviceType] = None,
):
    """Per-timestep traces for MULTIPLE modes at once (encodes the windows ONCE).

    Returns ``{mode: np.ndarray[T]}``. Used to record every variant's cost on each generated
    trajectory for comparison, independent of which (if any) variant is enforced. ``target_overrides``
    optionally maps mode -> target (else each mode's default).
    """
    pack, pos, tensor_args = _prep_segment(position, source_dt, checkpoint_path, n_joints, tensor_args)
    T = pos.shape[0]
    target_overrides = target_overrides or {}
    out = {}
    with torch.no_grad():
        wins, _, nw, starts, n2 = positions_to_windows(pos.unsqueeze(0), source_dt, pack)
        mu, logvar = pack["model"].encode(wins)  # encode ONCE, score every mode from it
        for mode in modes:
            scores = score_from_latents(mu, logvar, wins, pack, mode, target_overrides.get(mode))
            out[mode] = _scatter_interp(scores, starts, n2, pack["window"], T, tensor_args)
    return out
