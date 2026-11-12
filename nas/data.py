"""CIFAR-10 kept entirely on the GPU as uint8, with GPU-side augmentation (fast for tiny proxy runs)."""
import os

import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(root):
    """Loads CIFAR-10 from the HF parquet files (uoft-cs/cifar10) and caches uint8 tensors."""
    cache = os.path.join(root, "cifar10_uint8.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    d = {}
    for split in ("train", "test"):
        x, y = _read_parquet(os.path.join(root, f"{split}.parquet"))
        d[f"{split}_x"], d[f"{split}_y"] = x, y
    torch.save(d, cache)
    return d


def _read_parquet(path):
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Download: https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
            f"{os.path.basename(path).replace('.parquet', '')}-00000-of-00001.parquet"
        )
    t = pq.read_table(path).to_pydict()
    imgs = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(t["label"], dtype=torch.long)
    return x, y


def get_splits(root, val_size=5000, split_seed=0):
    """Fixed train/val split of the official train set. The test set is untouched during search."""
    d = load_cifar10(root)
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(d["train_x"]), generator=g)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    return {
        "train_x": d["train_x"][tr_idx], "train_y": d["train_y"][tr_idx],
        "val_x": d["train_x"][val_idx], "val_y": d["train_y"][val_idx],
        "test_x": d["test_x"], "test_y": d["test_y"],
    }


def normalize(xb_uint8):
    x = xb_uint8.float().div_(255.0)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x, gen, pad=4):
    """Per-sample random crop (reflect pad) + horizontal flip, all on device."""
    b, c, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None, :])
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])
    bi = torch.arange(b, device=x.device)[:, None, None, None]
    ci = torch.arange(c, device=x.device)[None, :, None, None]
    x = xp[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(b, device=x.device, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


class GPULoader:
    def __init__(self, x, y, batch_size, shuffle, aug, device, seed=0, drop_last=False):
        self.x = x.to(device)
        self.y = y.to(device)
        self.bs = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.drop_last = drop_last
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def __len__(self):
        n = len(self.x)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, device=self.x.device, generator=self.gen) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(len(self)):
            j = idx[i * self.bs:(i + 1) * self.bs]
            xb = normalize(self.x[j])
            if self.aug:
                xb = augment(xb, self.gen)
            yield xb, self.y[j]
