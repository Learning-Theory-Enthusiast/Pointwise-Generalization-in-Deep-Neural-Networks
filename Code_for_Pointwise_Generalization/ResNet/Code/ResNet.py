import os
import math
import csv
import copy
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

# ---------------- ResNet (BasicBlock) ----------------
class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNet110(nn.Module):
    """
    Note:
      `num_blocks=[18,18,18]` creates a very deep network.
      If you intended a typical ResNet-20, use [3,3,3].
      If you intended something like ResNet-110, use [18,18,18].
    """
    def __init__(
        self,
        block: type[nn.Module] = BasicBlock,
        num_blocks: List[int] = [18, 18, 18],
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_planes = 16
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(
        self, block: type[nn.Module], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


# ---------------- training / accuracy / margin ----------------
def train_one_epoch(
    model: nn.Module, loader: DataLoader, opt: optim.Optimizer, crit: nn.Module, dev: torch.device
) -> None:
    """Run one standard supervised training epoch."""
    model.train()
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    """Compute top-1 accuracy on the provided loader."""
    model.eval()
    tot, corr = 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model(x).argmax(1)
        corr += (pred == y).sum().item()
        tot += y.size(0)
    return corr / tot


@torch.no_grad()
def compute_margin(
    model: nn.Module,
    loader: DataLoader,
    dev: torch.device,
    percentile: float = 0.05,
    eps: float = 1e-12,
) -> float:
    """
    Estimate the classification margin as the `percentile` of (logit_y - max logit_other)
    across the dataset. Lower-bounded by `eps` for numerical stability.
    """
    model.eval()
    margins: List[torch.Tensor] = []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        idx = torch.arange(len(y), device=dev)
        corr = logits[idx, y]
        logits[idx, y] = -float("inf")
        other = logits.max(1).values
        margins.append(corr - other)
    if not margins:
        return eps
    m = torch.cat(margins)
    k = max(1, int(percentile * m.numel()))
    return max(m.kthvalue(k).values.item(), eps)


# ---------------- spectral / Frobenius norms ----------------
@torch.no_grad()
def spectral_norm_linear(weight: torch.Tensor) -> float:
    W = weight.detach().double().cpu()
    return float(torch.linalg.svdvals(W).max().item())


@torch.no_grad()
def spectral_norm_conv(conv_w: torch.Tensor) -> float:
    W = conv_w.detach().double().view(conv_w.size(0), -1).cpu()
    return float(torch.linalg.svdvals(W).max().item())


@torch.no_grad()
def frob_norm2_conv(conv_w: torch.Tensor) -> float:
    return float(conv_w.detach().double().pow(2).sum().item())


# ---------------- precompute statistics for ResNet ----------------
@torch.no_grad()
def precompute_stats_resnet(
    model: nn.Module,
    loader: DataLoader,
    dev: torch.device,
    include_skip_in_C: bool = True,
    treat_skip: str = "sum",
    sketch_max: int = 8192,
    sketch_frac: int = 8,
) -> Tuple[int, float, List[float], List[np.ndarray], List[int]]:
    """
    Collect quantities needed by the sum-form bound:

      - C: sum of squared Frobenius norms over weights
      - suffix: product of squared spectral norms from layer l+1 to L
      - evs_scaled_list: sketched covariance eigenvalues per feature block, each scaled by L * suffix[l+1]
      - d_list: output dimensions for conceptual layers (used in the bound)

    Sketching method:
      For each feature block F (flattened to [B, d]), project with a Gaussian matrix P ∈ R^{d×r}
      where r = min(sketch_max, d // sketch_frac) and accumulate ZᵀZ with Z = FP.
      The eigenvalues of (ZᵀZ / total) approximate those of Pᵀ Σ P.

    Args:
      include_skip_in_C: include downsample (shortcut) conv in Frobenius energy C
      treat_skip: "ignore" uses only main-branch conv2 spectral term; any other value sums with skip
      sketch_max: cap for sketch dimension r
      sketch_frac: set r as d // sketch_frac when smaller than sketch_max

    Returns:
      n, C, suffix, evs_scaled_list, d_list
    """
    model.eval()
    n = len(loader.dataset)
    spec_sq: List[float] = []
    d_list: List[int] = []
    C = 0.0

    # Stem: conv1
    s1 = spectral_norm_conv(model.conv1.weight)
    spec_sq.append(s1**2)
    d_list.append(model.conv1.out_channels)
    C += frob_norm2_conv(model.conv1.weight)

    # Residual blocks
    def add_basicblock_stats(seq: nn.Sequential) -> None:
        nonlocal C
        for blk in seq:
            assert isinstance(blk, BasicBlock)

            # conv1
            s_c1 = spectral_norm_conv(blk.conv1.weight)
            spec_sq.append(s_c1**2)
            d_list.append(blk.conv1.out_channels)
            C += frob_norm2_conv(blk.conv1.weight)

            # conv2
            s_c2 = spectral_norm_conv(blk.conv2.weight)
            s_skip = 1.0
            if isinstance(blk.shortcut, nn.Sequential) and len(blk.shortcut) > 0:
                conv_ds = next(m for m in blk.shortcut.modules() if isinstance(m, nn.Conv2d))
                s_skip = spectral_norm_conv(conv_ds.weight)
                if include_skip_in_C:
                    C += frob_norm2_conv(conv_ds.weight)

            # Combine spectral contributions for residual addition
            s_merge = s_c2**2 if treat_skip == "ignore" else (s_c2**2 + s_skip**2)
            spec_sq.append(s_merge)
            d_list.append(blk.conv2.out_channels)
            C += frob_norm2_conv(blk.conv2.weight)

    add_basicblock_stats(model.layer1)
    add_basicblock_stats(model.layer2)
    add_basicblock_stats(model.layer3)

    # Head: global average pool -> fully-connected
    s_fc = spectral_norm_linear(model.linear.weight)
    spec_sq.append(s_fc**2)
    d_list.append(model.linear.out_features)
    C += float(model.linear.weight.detach().norm("fro").pow(2).item())

    # Suffix product of squared spectral norms
    L = len(spec_sq)
    suffix = [1.0] * (L + 1)
    for j in range(L - 1, -1, -1):
        suffix[j] = suffix[j + 1] * spec_sq[j]

    # ---------- Capture inputs and post-ReLU activations ----------
    feats: List[torch.Tensor] = []

    def _save_input(_m, inp):
        feats.append(inp[0].detach())
        return None

    hooks = [model.register_forward_pre_hook(_save_input)]
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(lambda _m, _i, o: feats.append(o.detach())))

    def _replace_with_pool(_m, _i, o):
        # If a pooling layer follows a ReLU, replace the last feature block with the pooled output
        if feats:
            feats[-1] = o.detach()

    for m in model.modules():
        if isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
            hooks.append(m.register_forward_hook(_replace_with_pool))

    # ---------- Gaussian sketch: accumulate Gram matrices on r×r ----------
    small_grams: List[torch.Tensor] = []  # each is r×r
    P_list: List[torch.Tensor] = []       # P ∈ R^{d×r}
    total = 0

    for xb, _ in loader:
        xb = xb.to(dev, non_blocking=True)
        b = xb.size(0)
        total += b
        feats.clear()
        _ = model(xb)  # triggers hooks; fills `feats`

        # Grow holders the first time we see more feature blocks
        if len(small_grams) < len(feats):
            small_grams.extend([None] * (len(feats) - len(small_grams)))
            P_list.extend([None] * (len(feats) - len(P_list)))

        for j, F in enumerate(feats):
            F = F.to(dev)
            F_flat = F.view(b, -1)
            d = F_flat.size(1)

            # Initialize/refresh Gaussian projection P for this layer if shape changed
            if P_list[j] is None or P_list[j].size(0) != d:
                r = min(sketch_max, d // sketch_frac)
                P_list[j] = torch.randn(d, r, device=F_flat.device) / math.sqrt(r)
            P = P_list[j]

            Z = F_flat @ P
            Gs = Z.t() @ Z
            small_grams[j] = Gs if small_grams[j] is None else small_grams[j] + Gs

    for h in hooks:
        h.remove()

    # ---------- Eigenvalues of sketched covariances, scaled by S_l ----------
    evs_scaled_list: List[np.ndarray] = []
    for j, Bsum in enumerate(small_grams):
        if Bsum is None:
            evs_scaled_list.append(np.array([], dtype=np.float64))
            continue
        B = (Bsum / total).float()
        ev = torch.linalg.eigvalsh(B)
        ev.clamp_min_(0.0)
        S_l = L * (suffix[j + 1] if (j + 1) < len(suffix) else 1.0)
        evs_scaled_list.append((ev * S_l).cpu().numpy() if ev.numel()
                               else np.array([], dtype=np.float64))
    return n, C, suffix, evs_scaled_list, d_list


# ---------------- Bound (sum form; returns per-layer effective counts and raw sum) ----------------
def bound_for_eps(
    eps: float,
    n: int,
    C: float,
    evs_scaled_list: List[np.ndarray],
    d_list: List[int],
) -> Tuple[float, List[int], float]:
    """
    Compute the sum-form bound term at a given epsilon.

    Returns:
      sum_no_eps: sqrt( sum_j d_j * sum_{k: ev>=thr} log(ev/thr) / n )
      eff_counts_per_layer: number of eigenvalues above threshold per layer
      sum_terms: raw unnormalized sum ∑_j d_j * ∑_k log(ev/thr)
    """
    thr = (eps ** 2) / C
    terms: List[float] = []
    eff_counts_per_layer: List[int] = []
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
    sum_no_eps = math.sqrt(arr.sum() / n)
    sum_terms = float(arr.sum())  # Riemannian dimension complexity
    return sum_no_eps, eff_counts_per_layer, sum_terms


@torch.no_grad()
def tune_epsilon_min_from_stats(
    n: int,
    C: float,
    evs_scaled_list: List[np.ndarray],
    d_list: List[int],
    iters: int = 60,
) -> Tuple[float, float, List[int], float]:
    """
    Ternary-search epsilon to minimize: eps + sum_term(eps).

    Returns:
      eps_star: optimizer of the objective
      bound_val: eps_star + sum_term(eps_star)
      eff_counts: per-layer effective eigenvalue counts at eps_star
      sum_terms: raw sum at eps_star
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
    sb, eff_counts, sum_terms = bound_for_eps(eps_star, n, C, evs_scaled_list, d_list)
    return eps_star, eps_star + sb, eff_counts, sum_terms


# ---------------- Main (CIFAR-10 + ResNet) ----------------
def main() -> None:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- Transforms ----------
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),            # Common CIFAR-10 augmentation
        transforms.RandomHorizontalFlip(),               # Common CIFAR-10 augmentation
        transforms.ToTensor(),
        transforms.Normalize(                            # CIFAR-10 mean/std
            mean=[0.4914, 0.4822, 0.4465],
            std =[0.2023, 0.1994, 0.2010],
        ),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(                            # CIFAR-10 mean/std
            mean=[0.4914, 0.4822, 0.4465],
            std =[0.2023, 0.1994, 0.2010],
        ),
    ])

    # ---------- Datasets & loaders ----------
    tr_ds = datasets.CIFAR10('data', train=True,  transform=tf_train, download=True)
    te_ds = datasets.CIFAR10('data', train=False, transform=tf_test,  download=True)

    tr_ldr = DataLoader(tr_ds, batch_size=128, shuffle=True,  num_workers=8, pin_memory=True)
    te_ldr = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    model = ResNet110().to(dev)

    # ---------- Optimizer & schedule ----------
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # Typical CIFAR-10 schedule; adjust milestones as needed
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 150, 200], gamma=0.1)
    crit = nn.CrossEntropyLoss()

    n = len(tr_ds)  # CIFAR-10 training set size (50,000)

    # ---------- Logging filenames ----------
    effdim_path = 'eff_dims_resnet110.csv'  # retained filename for compatibility
    eff_header_written = False

    results: List[List[float]] = []
    for epoch in range(1, 251):
        train_one_epoch(model, tr_ldr, opt, crit, dev)
        sched.step()

        tr_err = 1 - accuracy(model, tr_ldr, dev)
        te_err = 1 - accuracy(model, te_ldr, dev)
        gap = te_err - tr_err

        n_chk, C, suffix, evs_scaled_list, d_list = precompute_stats_resnet(
            model, tr_ldr, dev, include_skip_in_C=True, treat_skip="sum"
        )
        assert n_chk == n

        eps_sum, sum_bd, eff_counts, riemann_sum_terms = tune_epsilon_min_from_stats(
            n, C, evs_scaled_list, d_list, iters=50
        )

        # Write header once, then append per epoch
        if not eff_header_written:
            with open(effdim_path, 'w', newline='') as f:
                csv.writer(f).writerow(['epoch'] + [f'layer{j + 1}' for j in range(len(eff_counts))])
            eff_header_written = True
        with open(effdim_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch] + eff_counts)

        results.append([epoch, tr_err, te_err, gap, eps_sum, sum_bd, riemann_sum_terms])

        print(
            f"epoch={epoch:3d} "
            f"train_err={tr_err:.4f} test_err={te_err:.4f} gap={gap:.4f} "
            f"eps_sum={eps_sum:.4e} sum_bd={sum_bd:.4e} "
            f"riemann_sum_terms={riemann_sum_terms:.4e} "
        )

    # Dump per-epoch summary table
    with open('bounds_resnet110.csv', 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'train_err', 'test_err', 'gap', 'epsilon_sum', 'sum_bound', 'riemann_sum_terms'
        ])
        csv.writer(f).writerows(results)


if __name__ == "__main__":
    main()
