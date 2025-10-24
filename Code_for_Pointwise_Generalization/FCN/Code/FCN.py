import os
import math
import csv
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple

# Import measures.py from the same directory
import measures as cm


# ============================== 1) Configurable deep FCN ==============================
class DeepFCN(nn.Module):
    """
    A configurable fully-connected network.

    Architecture: [Linear, ReLU] * (#hidden layers) + Linear(output)

    You can specify either:
      - `hidden_dims`: a list of hidden layer widths (preferred), or
      - (`depth`, `width`): number of hidden layers and a shared width.

    The total number of linear layers is L = (#hidden layers) + 1 (for the output layer).
    The number of ReLU layers equals the number of hidden layers.
    """
    def __init__(
        self,
        depth: int | None = None,
        width: int = 64,
        in_dim: int = 28 * 28,
        num_classes: int = 10,
        hidden_dims: List[int] | None = None,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d_in = in_dim

        # Build hidden stack either from `hidden_dims` or from (depth, width)
        if hidden_dims is not None:
            for h in hidden_dims:
                layers += [nn.Linear(d_in, h), nn.ReLU(inplace=True)]
                d_in = h
        else:
            assert depth is not None, "Either `hidden_dims` or `depth` must be provided."
            for _ in range(depth):
                layers += [nn.Linear(d_in, width), nn.ReLU(inplace=True)]
                d_in = width

        # Final classifier layer (no activation)
        layers.append(nn.Linear(d_in, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten images to vectors before passing through the FCN
        x = x.view(x.size(0), -1)
        return self.net(x)


# ============================== 2) Training & evaluation ==============================
def train_one_epoch(model: nn.Module, loader: DataLoader, opt: optim.Optimizer,
                    crit: nn.Module, dev: torch.device) -> None:
    """One standard supervised training epoch."""
    model.train()
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    """Compute top-1 accuracy."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def compute_margin(model: nn.Module, loader: DataLoader, dev: torch.device,
                   percentile: float = 0.05, eps: float = 1e-12) -> float:
    """
    Estimate the margin as the percentile of (logit_y - max logit_other) over the dataset.
    Returns a small positive floor `eps` if empty.
    """
    model.eval()
    margins: list[torch.Tensor] = []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        idx = torch.arange(len(y), device=dev)
        corr = logits[idx, y]
        logits[idx, y] = -float('inf')
        other = logits.max(1).values
        margins.append(corr - other)
    if not margins:
        return eps
    m = torch.cat(margins)
    k = max(1, int(percentile * m.numel()))
    return max(m.kthvalue(k).values.item(), eps)


# ============================== 3) Spectral stats & feature covariances ==============================
@torch.no_grad()
def precompute_stats(model: nn.Module, loader: DataLoader, dev: torch.device):
    """
    Compute quantities used by the Riemannian-dimension-style bound.

    Returns:
      n:                    number of training samples
      C:                    sum_{l=1}^L ||W_l||_F^2  (Frobenius energy over linear layers)
      suffix:               length-(L+1) array, suffix[l] = ∏_{k=l+1}^L ||W_k||_σ^2, with suffix[L] = 1
      evs_list[j]:          eigenvalues of Σ_{l-1} scaled by S_l (i.e., suffix[j+1]*L * λ_k)
      d_list[j]:            output dimension d_l of linear layer l
    """
    model.eval()
    n = len(loader.dataset)

    # ---- Gather linear-layer weights ----
    wmods = [m for m in model.modules() if isinstance(m, nn.Linear)]
    L = len(wmods)
    mats = [m.weight.detach().to(dev, dtype=torch.float64) for m in wmods]

    # Frobenius energy and layer output dimensions
    C = float(sum((W.norm('fro') ** 2).item() for W in mats))
    d_list = [W.size(0) for W in mats]

    # ---- Build suffix products of spectral norms (squared) as an upper bound scale ----
    sig = [float(torch.linalg.svdvals(W).max().item()) for W in mats]
    suffix = [1.0] * (L + 1)
    for j in range(L - 1, -1, -1):
        suffix[j] = suffix[j + 1] * (sig[j] ** 2)

    # ---- Capture inputs and post-ReLU activations via hooks ----
    acts = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    feats: list[torch.Tensor | None] = [None] * (len(acts) + 1)
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _save_input(_m, inp):
        x0 = inp[0]
        feats[0] = x0.view(x0.size(0), -1).detach().cpu()
        return None

    hooks.append(model.register_forward_pre_hook(_save_input))
    for i, act_layer in enumerate(acts, 1):
        # Use default arg `idx=i` to bind loop index in closure
        hooks.append(
            act_layer.register_forward_hook(
                lambda _m, _i, o, idx=i: feats.__setitem__(idx, o.detach().cpu())
            )
        )

    # Accumulate (unnormalized) covariance sums across the dataset for each recorded feature block
    covs: list[torch.Tensor | None] = [None] * len(feats)
    total = 0
    for x, _ in loader:
        for j in range(len(feats)):
            feats[j] = None
        b = x.size(0)
        total += b
        _ = model(x.to(dev))  # populates feats[] via hooks
        for j, F in enumerate(feats):
            if F is None:
                continue
            F_flat = F.view(b, -1).to(dev, non_blocking=True)
            cov = F_flat.T @ F_flat
            covs[j] = cov if covs[j] is None else covs[j] + cov

    # Remove hooks to avoid leaking references
    for h in hooks:
        h.remove()

    # EVD for each covariance and scale by suffix[j+1]*L
    evs_list: list[np.ndarray] = []
    for j, Gsum in enumerate(covs):
        if Gsum is None:
            evs = np.array([])
        else:
            Gi = (Gsum / total).double().cpu()
            ev = torch.linalg.eigvalsh(Gi).numpy()
            evs = np.maximum(ev, 0)
            scale = suffix[j + 1] * L
            evs = evs * scale
        evs_list.append(evs)

    return n, C, suffix, evs_list, d_list


# ============================== 4) Bound (sum-style) ==============================
def bound_for_eps(eps: float, n: int, C: float,
                  evs_scaled_list: List[np.ndarray], d_list: List[int]) -> tuple[float, list[int], float]:
    """
    Compute the sum-form term of the bound at a given epsilon.
    Returns:
      sum_no_eps: sqrt( sum_j d_j * sum_{k: ev>=thr} log(ev/thr) / n )
      eff_counts_per_layer: number of kept eigenvalues per layer (ev >= thr)
      sum_terms: the raw sum ∑_j d_j * ∑_k log(ev/thr) before normalization
    """
    thr = (eps ** 2) / C
    terms: list[float] = []
    eff_counts_per_layer: list[int] = []
    for j in range(len(evs_scaled_list)):
        d_l = d_list[j]
        evs = evs_scaled_list[j]
        if evs.size == 0:
            eff_counts_per_layer.append(0)
            terms.append(0.0)
            continue
        sel = evs[evs >= thr]
        eff_counts_per_layer.append(int(sel.size))
        if sel.size > 0:
            s = np.log(sel / thr).sum()
            terms.append(d_l * s)
        else:
            terms.append(0.0)
    arr = np.array(terms, dtype=np.float64)
    sum_no_eps = math.sqrt(max(arr.sum(), 0.0) / n)
    sum_terms = float(arr.sum())
    return sum_no_eps, eff_counts_per_layer, sum_terms


@torch.no_grad()
def tune_epsilon_min_from_stats(
    n: int,
    C: float,
    evs_scaled_list: List[np.ndarray],
    d_list: List[int],
    iters: int = 50,
) -> tuple[float, float, list[int], float]:
    """
    Minimize eps + sum_term(eps) via ternary search over a loose range.

    Returns:
      eps_star:           argmin epsilon
      bound_val:          eps_star + sum_term(eps_star)
      eff_counts_per_layer
      sum_terms_at_eps
    """
    eps_lo = math.sqrt(1.0 / n)
    max_eig = max((ev.max() if ev.size else 0.0) for ev in evs_scaled_list)
    eps_hi = math.sqrt(C * max_eig)

    for _ in range(iters):
        m1 = eps_lo + (eps_hi - eps_lo) / 3.0
        m2 = eps_hi - (eps_hi - eps_lo) / 3.0
        sb1, _, _ = bound_for_eps(m1, n, C, evs_scaled_list, d_list)
        sb2, _, _ = bound_for_eps(m2, n, C, evs_scaled_list, d_list)
        if m1 + sb1 > m2 + sb2:
            eps_lo = m1
        else:
            eps_hi = m2

    eps_star = (eps_lo + eps_hi) / 2.0
    sb, eff_counts_per_layer, sum_terms_at_eps = bound_for_eps(
        eps_star, n, C, evs_scaled_list, d_list
    )
    bound_val = eps_star + sb
    return eps_star, bound_val, eff_counts_per_layer, sum_terms_at_eps


# ============================== 5) Main experiment loop ==============================
def main() -> None:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr_ds = datasets.MNIST('data', train=True, transform=tf, download=True)
    te_ds = datasets.MNIST('data', train=False, transform=tf, download=True)

    # Dataloaders: use pinned memory + prefetch for better throughput
    tr_ldr = DataLoader(
        tr_ds, batch_size=128, shuffle=True,
        num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    te_ldr = DataLoader(
        te_ds, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    # Seven FCN configs (each with 9 hidden layers):
    tail_widths = [64, 128, 256, 512, 1024, 2048, 4096]
    configs = [[2048, 2048] + [w] * 7 for w in tail_widths]

    effdim_path_tpl = 'eff_dims_fcn_cfg{:02d}.csv'
    result_path_tpl = 'bounds_fcn_cfg{:02d}.csv'

    for cfg_id, hidden_dims in enumerate(configs, 1):
        d = len(hidden_dims)
        print(f"\n=== FCN cfg#{cfg_id} (hidden_dims={hidden_dims}) ===")
        results: list[list[float | int]] = []
        eff_header_written = False

        # Build model per configuration; keep a copy of the init state for baseline measures
        model = DeepFCN(hidden_dims=hidden_dims).to(dev)
        init_model = copy.deepcopy(model)

        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 170], gamma=0.1)
        crit = nn.CrossEntropyLoss()

        m = len(tr_ds)

        for epoch in range(1, 201):
            train_one_epoch(model, tr_ldr, opt, crit, dev)
            sched.step()

            tr_err = 1 - accuracy(model, tr_ldr, dev)
            te_err = 1 - accuracy(model, te_ldr, dev)
            gap = te_err - tr_err

            mrg = compute_margin(model, tr_ldr, dev)

            # Precompute stats for bound
            n, C, suffix, evs_scaled_list, d_list = precompute_stats(model, tr_ldr, dev)

            # Ternary-search epsilon for the sum-style bound
            eps_sum, sum_bd, eff_counts_per_layer, riemann_sum_terms = tune_epsilon_min_from_stats(
                n, C, evs_scaled_list, d_list, iters=500
            )

            # Compare against Bartlett et al. (2017) Spec L_{2,1} bound
            measures, bounds = cm.calculate(
                trained_model=model,
                init_model=init_model,
                device=dev,
                train_loader=tr_ldr,
                margin=mrg,
                nchannels=1, nclasses=10, img_dim=28
            )
            key = 'Spec_L_{2,1} Bound (Bartlett et al. 2017)'
            spec_l21_div_sqrtm = bounds[key]
            spec_l21_no_div = (spec_l21_div_sqrtm * math.sqrt(m)) ** 2

            # Write effective counts CSV once, then append per epoch
            effdim_path = effdim_path_tpl.format(cfg_id)
            if not eff_header_written:
                with open(effdim_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch'] + [f'layer{j + 1}' for j in range(len(eff_counts_per_layer))])
                eff_header_written = True
            with open(effdim_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch] + eff_counts_per_layer)

            results.append([
                cfg_id, epoch, tr_err, te_err, gap,
                eps_sum, sum_bd, riemann_sum_terms,
                spec_l21_no_div
            ])

            print("effective =", eff_counts_per_layer)
            print(
                f"cfg#{cfg_id:02d} depth={d:2d} epoch={epoch:3d} "
                f"train_err={tr_err:.4f} test_err={te_err:.4f} gap={gap:.4f} "
                f"eps_sum={eps_sum:.4e} sum_bd={sum_bd:.4e} "
                f"riemann_sum_terms={riemann_sum_terms:.4e} "
                f"Spec_L21={spec_l21_no_div:.4e}"
            )

        # Dump per-config result table
        result_csv = result_path_tpl.format(cfg_id)
        header = [
            'cfg_id', 'epoch', 'train_err', 'test_err', 'gap',
            'epsilon_sum', 'sum_bound', 'riemann_sum_terms',
            'Spec_L21'
        ]
        with open(result_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
        results.clear()


if __name__ == '__main__':
    main()
