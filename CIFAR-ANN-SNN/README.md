# TinyCifarNet — ANN for CIFAR-10 (ANN→SNN)

PyTorch CNN with **28 626** folded weights/biases (limit ≈ 30 000), ReLU + AvgPool.
BatchNorm is used in training and folded into Conv at export.

**Test accuracy: 83.94%** (CIFAR-10, valid conv, no `pool3`, 80 epochs, seed 42). Previous padded recipe was 83.88%.

## Layout

| Stage | Ops | Spatial |
|------|-----|---------|
| Block 1 | Conv 3→16, ReLU, Conv 16→16, ReLU, AvgPool2 | 32→30→28→14 |
| Block 2 | Conv 16→32, ReLU, Conv 32→32, ReLU, AvgPool2 | 14→12→10→5 |
| Block 3 | Conv 32→40, ReLU, AdaptiveAvgPool | 5→3→1 |
| Head | Linear 40→10 | logits |

All convolutions are **valid** (`padding=0`). There is no `pool3` (2×2 on 3×3 would not tile). GAP over 3×3 keeps a 40-D vector for CoLaNET.

## Setup

```bash
pip install -r requirements.txt
```

Uses local `CIFAR/Workplace/CIFAR10.bin` + `CIFAR10.target.txt` (CHW uint8) when present; otherwise downloads via torchvision.

## Train

```bash
python train.py --epochs 80
```

Best checkpoint and exports go to `artifacts/`.

## Artifacts (trained)

| File | Contents |
|------|----------|
| `artifacts/architecture.json` | Layer graph, shapes, preprocessing, metrics |
| `artifacts/weights.npz` | Folded tensors: `conv*.weight/bias`, `fc.weight/bias` |
| `artifacts/weights_dump.txt` | Same tensors as flat text |
| `artifacts/best.pt` | PyTorch checkpoint (with BN, for resume) |
| `artifacts/metrics.json` | Test accuracy, param counts |
| `artifacts/pre_fc_activations.csv` | 60k × 40 flatten GAP (CoLaNET) |
| `artifacts/activations/<layer>.npy` | NCHW float32 maps for `conv2`…`gap` (not `conv1`) |

Dump layer maps (needs `best.pt` and `CIFAR10.bin`):

```bash
python extract_pre_fc_activations.py
```

Writes `artifacts/pre_fc_activations.csv` and `artifacts/activations/{conv2,pool1,conv3,conv4,pool2,conv5,gap}.npy` (CIFAR file order, 60 000 images). `conv1` is omitted: the SNN computes it in `fromFile`.

Load weights:

```python
import numpy as np, json
arch = json.load(open("artifacts/architecture.json"))
w = np.load("artifacts/weights.npz")
print(arch["layers"])
print({k: w[k].shape for k in w.files})
```

## Design notes

- ReLU + average pooling for straightforward rate-based / exact ReLU→spike mapping.
- BN only while training; export absorbs it into Conv (+bias).
- No residual connections; ArgMax on logits for class id (no Softmax in the exported graph).
- Tooling: **PyTorch** — standard for small CNNs, BN folding, and portable weight export.
