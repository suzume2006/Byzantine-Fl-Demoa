# DeMoA on MNIST: my small reproduction

I read the paper **"Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation"** (Otsuka, Takezawa, Yamada, arXiv:2509.02970) and tried to re-implement its main idea on MNIST. This is a learning project, **not** the authors' code.

## The problem, in simple words

In federated learning, many devices train one shared model and send updates to a server. Some devices may be broken or malicious (called *Byzantine* clients). The server defends itself with tools like the median, which only work if honest devices are the majority.

But the server usually hears from only a few devices each round. By bad luck, those few can be mostly malicious, even if malicious devices are only 20% overall. In that round the server gets fooled, and one fooled round can ruin the model.

## The idea (DeMoA)

The server keeps the **last update (momentum) of every device**. Each round it:

1. updates the notes of the devices that spoke,
2. slightly fades the notes of the silent devices,
3. combines **all** the notes with a robust aggregator (median, etc.),
4. updates the model.

So malicious devices are always the true 20%, never a lucky majority.

## What's in this repo

- `demoa_mnist.py`: the whole implementation (one file)
- `logs/` and `results/`: the output of my runs (or the loose `.txt` and `.json` files, if you didn't make folders)

It compares three methods:

| Method | What it does |
|---|---|
| `fedavg` | averages only the devices that spoke this round |
| `fedcm` | same, but with momentum |
| `demoa` | keeps every device's last momentum and uses all of them |

Aggregators: `mean`, `cm` (median), `cclip` (centered clipping).
Attacks: `none`, `lf` (label flip), `bf` (bit flip), `ipm`, `alie`.

## How to run it

```bash
pip install torch numpy matplotlib
python demoa_mnist.py --p 0.5 --attack none --agg mean --rounds 60 --eval_every 10
```

MNIST downloads by itself. `matplotlib` is optional (it is only for the plot). Results are always saved as a `.json` file.

More examples:

```bash
# strong attack, median aggregator, only 10% of devices per round
python demoa_mnist.py --p 0.1 --attack ipm --ipm_eps 10 --agg cm --rounds 300

# harder case: non-IID data with bucketing
python demoa_mnist.py --p 0.1 --attack alie --agg cclip --noniid --bucket 2 --rounds 300
```

## What I found

Test accuracy (%) after 300 rounds, with 25 devices and 5 malicious. Each row is **one run**, so please don't treat these as exact numbers.

| Setting | fedavg | fedcm | demoa |
|---|---|---|---|
| No attack, p = 0.5 (60 rounds) | 90.7 | 91.0 | 91.4 |
| IPM attack, median, p = 0.1, seed 0 | 10.1 | 9.8 | 89.3 |
| IPM attack, median, p = 0.1, seed 1 | 9.8 | 9.8 | 89.2 |
| IPM attack, median, p = 0.05, seed 0 | 9.8 | 9.8 | 90.5 |
| Label flipping, median, p = 0.1 | 83.1 | 76.3 | 91.9 |
| ALIE, clipping, p = 0.1, IID | 96.3 | 96.4 | 95.8 |
| ALIE, clipping, p = 0.1, non-IID | 89.7 | 91.6 | 88.1 |

In plain words:

- **With no attack,** all three methods work fine, so the code is behaving.
- **With the strong IPM attack,** `fedavg` and `fedcm` fell to about 10% (basically guessing) in all three runs, while `demoa` stayed around 89 to 90%. This is the effect the paper talks about.
- **With label flipping,** `demoa` did best, but the curves are bumpy and I only ran one seed.
- **With ALIE, I did not reproduce the paper's result.** The baselines did fine there and were as good as, or better than, `demoa`. My version of this attack is probably too mild.

## What's different from the paper (honest list)

- My attacks change the malicious device's **gradient** and then apply the normal momentum rule. In the paper, a malicious device can send anything.
- No Krum or RFA aggregators.
- Only MNIST (no CIFAR-10 / ResNet-18).
- One seed per run and no learning-rate tuning, so the numbers won't match the paper exactly.

## Limits and next steps

- The server stores one vector per device, which gets expensive with millions of devices.
- To do: run more seeds, attack the momentum directly, add Krum and RFA, tune the learning rate.

## Credit

All credit for the method goes to the paper's authors. Any mistakes in this re-implementation are mine.
