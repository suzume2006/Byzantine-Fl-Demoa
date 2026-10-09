"""
DeMoA (Delayed Momentum Aggregation) - small MNIST re-implementation.

Compares three methods under Byzantine attacks + partial participation:
  fedavg : aggregate only the sampled clients' raw gradients
  fedcm  : client momentum, aggregate only the sampled clients
  demoa  : server caches momentum of ALL clients, fades silent ones by (1 - alpha*p),
           and aggregates over ALL n clients every round   <-- the paper's idea

Simplifications vs the paper (read before comparing numbers):
  * Attacks are applied to the Byzantine client's *gradient*, then the normal momentum
    rule is applied. A real Byzantine client could send anything as its momentum.
  * Attack statistics (mean/std of honest clients) come from the server's cached honest rows.
  * No Krum / RFA (only mean, coordinate-wise median, centered clipping).
  * 1 seed by default, 300 rounds (= "10 epochs" in the paper's MNIST setup).

Usage (Colab):
  !python demoa_mnist.py --p 0.1 --attack alie --agg cclip
  !python demoa_mnist.py --p 0.1 --attack lf --agg cm --noniid --bucket 2
"""
import argparse
import json
import math
from statistics import NormalDist

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import gzip
import os
import urllib.request

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------- model
class Net(nn.Module):
    """CONV-CONV-DROPOUT-FC-DROPOUT-FC (the MNIST net described in Appendix B)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.d1 = nn.Dropout(0.25)
        self.f1 = nn.Linear(9216, 128)
        self.d2 = nn.Dropout(0.5)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = self.d1(x).flatten(1)
        x = self.d2(F.relu(self.f1(x)))
        return self.f2(x)


# ----------------------------------------------------------------- data
MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _fetch(fname):
    os.makedirs("./data", exist_ok=True)
    path = os.path.join("./data", fname)
    if not os.path.exists(path):
        last = None
        for base in MNIST_MIRRORS:
            try:
                print(f"downloading {base}{fname}")
                urllib.request.urlretrieve(base + fname, path)
                last = None
                break
            except Exception as e:
                last = e
        if last is not None:
            raise RuntimeError(f"could not download {fname}: {last}")
    return path


def _read_idx(path, images):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    off = 16 if images else 8
    arr = np.frombuffer(buf, dtype=np.uint8, offset=off).copy()
    return arr.reshape(-1, 28, 28) if images else arr


def load_mnist():
    """Loads MNIST without torchvision."""
    def prep(xk, yk):
        X = torch.from_numpy(_read_idx(_fetch(MNIST_FILES[xk]), True)).float()
        X = X.div(255).sub(0.1307).div(0.3081).unsqueeze(1)
        Y = torch.from_numpy(_read_idx(_fetch(MNIST_FILES[yk]), False)).long()
        return X.to(dev), Y.to(dev)

    return prep("train_x", "train_y"), prep("test_x", "test_y")


def partition(Y, n, G, noniid, rng):
    """Clients 0..G-1 are honest, G..n-1 are Byzantine.
    IID: random equal split. Non-IID (Appendix B.4): sort by label, give each honest
    client one contiguous shard; Byzantine clients get random data."""
    N = len(Y)
    if not noniid:
        perm = rng.permutation(N)
        return [torch.tensor(s, device=dev) for s in np.array_split(perm, n)]
    order = np.argsort(Y.cpu().numpy(), kind="stable")
    shards = np.array_split(order, G)
    parts = [torch.tensor(s, device=dev) for s in shards]
    for _ in range(n - G):
        parts.append(torch.tensor(rng.choice(N, N // n, replace=False), device=dev))
    return parts


# ----------------------------------------------------------------- aggregators
def agg_mean(M, ref):
    return M.mean(0)


def agg_cm(M, ref):
    return M.median(0).values


def make_cclip(tau, L=3):
    def agg(M, ref):
        v = ref.clone()
        for _ in range(L):
            diff = M - v
            norm = diff.norm(dim=1, keepdim=True).clamp_min(1e-12)
            v = v + (diff * torch.clamp(tau / norm, max=1.0)).mean(0)
        return v
    return agg


def bucketize(M, s):
    """Random bucketing (Karimireddy et al. 2022): average within buckets of size s."""
    M = M[torch.randperm(M.shape[0], device=M.device)]
    return torch.stack([M[i:i + s].mean(0) for i in range(0, M.shape[0], s)])


# ----------------------------------------------------------------- training
def run(method, a):
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    (Xtr, Ytr), (Xte, Yte) = DATA
    n, nb = a.n, a.n_byz
    G = n - nb
    parts = partition(Ytr, n, G, a.noniid, rng)

    model = Net().to(dev)
    params = list(model.parameters())
    x = parameters_to_vector(params).detach().clone()
    d = x.numel()

    agg_fn = {"mean": agg_mean, "cm": agg_cm, "cclip": make_cclip(a.tau)}[a.agg]
    s_alie = math.floor(n / 2 + 1) - nb
    z_alie = NormalDist().inv_cdf((n - s_alie) / n)

    M = torch.zeros(n, d, device=dev)      # server-side cache: one momentum vector per client
    prev_agg = torch.zeros(d, device=dev)  # reference point for centered clipping

    def client_grad(i, flip=False):
        vector_to_parameters(x, params)
        idx = parts[i][torch.randint(len(parts[i]), (a.bs,), device=dev)]
        xb, yb = Xtr[idx], Ytr[idx]
        if flip:
            yb = 9 - yb
        model.train()
        model.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        return torch.cat([p.grad.reshape(-1) for p in params])

    def update_row(i, g, t):
        al = a.alpha
        if t == 0 or method == "fedavg":     # paper: alpha_1 = p_1 = 1
            M[i] = g
        elif method == "demoa":
            M[i] = (1 - al * a.p) * M[i] + al * g
        else:                                # fedcm
            M[i] = (1 - al) * M[i] + al * g

    def evaluate():
        vector_to_parameters(x, params)
        model.eval()
        with torch.no_grad():
            return (model(Xte).argmax(1) == Yte).float().mean().item() * 100

    accs = []
    for t in range(a.rounds):
        # ---- 1. sample clients (everyone participates in round 1)
        mask = torch.ones(n, dtype=torch.bool) if t == 0 else (torch.rand(n) < a.p)
        sampled = mask.nonzero().flatten().tolist()
        if len(sampled) == 0:
            accs.append(accs[-1] if accs else 10.0)
            continue

        # ---- 2. honest sampled clients compute momentum
        for i in sampled:
            if i < G:
                update_row(i, client_grad(i), t)

        # ---- 3. Byzantine sampled clients attack (omniscient: see honest rows)
        Mh = M[:G]
        mu, sd = Mh.mean(0), Mh.std(0)
        for i in sampled:
            if i >= G:
                if a.attack == "none":
                    g = client_grad(i)
                elif a.attack == "lf":
                    g = client_grad(i, flip=True)
                elif a.attack == "bf":
                    g = -client_grad(i)
                elif a.attack == "ipm":
                    g = -a.ipm_eps * mu
                elif a.attack == "alie":
                    g = mu - z_alie * sd
                update_row(i, g, t)

        # ---- 4. DeMoA only: server fades the cached momentum of silent clients itself
        if method == "demoa" and t > 0:
            silent = (~mask).to(dev)
            M[silent] *= (1 - a.alpha * a.p)

        # ---- 5. aggregate. DeMoA: ALL n rows.  Baselines: sampled rows only.
        rows = M if (method == "demoa" or t == 0) else M[sampled]
        if a.bucket > 1:
            rows = bucketize(rows, a.bucket)
        agg = agg_fn(rows, prev_agg)
        prev_agg = agg.clone()

        # ---- 6. model step
        x -= a.lr * agg

        if (t + 1) % a.eval_every == 0 or t == a.rounds - 1:
            acc = evaluate()
            accs.append(acc)
            print(f"[{method:6s}] round {t + 1:4d}  test acc {acc:5.1f}%", flush=True)
        else:
            accs.append(accs[-1] if accs else 10.0)
    return accs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["fedavg", "fedcm", "demoa"])
    ap.add_argument("--attack", default="alie", choices=["none", "lf", "bf", "ipm", "alie"])
    ap.add_argument("--agg", default="cclip", choices=["mean", "cm", "cclip"])
    ap.add_argument("--p", type=float, default=0.1, help="participation probability")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=10.0, help="clipping radius for cclip")
    ap.add_argument("--ipm_eps", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--n_byz", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=30)
    ap.add_argument("--noniid", action="store_true")
    ap.add_argument("--bucket", type=int, default=1, help="bucket size s (paper uses 2 for non-IID)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"device={dev}  args={vars(a)}")
    DATA = load_mnist()

    results = {}
    for m in a.methods:
        results[m] = run(m, a)

    tag = f"{a.agg}_{a.attack}_p{a.p}_{'noniid' if a.noniid else 'iid'}"
    with open(f"results_{tag}.json", "w") as f:
        json.dump(results, f)

    print(f"saved results_{tag}.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        for m, accs in results.items():
            plt.plot(range(1, len(accs) + 1), accs, label=m)
        plt.xlabel("round")
        plt.ylabel("test accuracy (%)")
        plt.title(f"{a.agg.upper()} / {a.attack} / p={a.p} / {'non-IID' if a.noniid else 'IID'}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plot_{tag}.png", dpi=150)
        print(f"saved plot_{tag}.png")
    except Exception as e:
        print(f"Plotting skipped ({e}). Accuracies are printed above and saved in the JSON file.")
