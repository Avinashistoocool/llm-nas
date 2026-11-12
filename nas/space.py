"""Hand-written seed architectures and a parameterised random search space (the random-search baseline).

Every architecture is plain PyTorch source defining `class Net(nn.Module)` with `__init__(self, num_classes=10)`,
so LLM candidates, seeds and random samples all go through the exact same validation/training path.
"""

SEEDS = {
    "simple_cnn": ("Plain 3-stage VGG-style CNN with BatchNorm and max-pooling.", '''
import torch
import torch.nn as nn


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(3, 32), conv_bn(32, 32), nn.MaxPool2d(2),
            conv_bn(32, 64), conv_bn(64, 64), nn.MaxPool2d(2),
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))
'''),
    "resnet_small": ("Small ResNet: 3 stages of basic residual blocks (widths 32/64/128), stride-2 downsampling.", '''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
        self.layer2 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64))
        self.layer3 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
'''),
    "mobile_small": ("MobileNetV2-style inverted residual blocks (expansion 4) with depthwise convolutions.", '''
import torch
import torch.nn as nn


class InvRes(nn.Module):
    def __init__(self, cin, cout, stride, expand=4):
        super().__init__()
        hid = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),
            nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),
            nn.Conv2d(hid, cout, 1, bias=False), nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        cfg = [(32, 1), (32, 1), (64, 2), (64, 1), (96, 2), (96, 1), (128, 1)]
        layers = [nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True)]
        cin = 32
        for cout, s in cfg:
            layers.append(InvRes(cin, cout, s))
            cin = cout
        layers += [nn.Conv2d(cin, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU6(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))
'''),
}


def seed_archs():
    return [(name, idea, code.lstrip()) for name, (idea, code) in SEEDS.items()]


_RANDOM_TEMPLATE = '''import torch
import torch.nn as nn
import torch.nn.functional as F


def act():
    return nn.{act}()


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(8, c // r), 1)
        self.fc2 = nn.Conv2d(max(8, c // r), c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))


class Block(nn.Module):
    def __init__(self, cin, cout, stride, kind, k, expand, use_se):
        super().__init__()
        p = k // 2
        if kind == "basic":
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, k, stride, p, bias=False), nn.BatchNorm2d(cout), act(),
                nn.Conv2d(cout, cout, k, 1, p, bias=False), nn.BatchNorm2d(cout))
        elif kind == "bottleneck":
            mid = max(8, cout // 2)
            self.body = nn.Sequential(
                nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), act(),
                nn.Conv2d(mid, mid, k, stride, p, bias=False), nn.BatchNorm2d(mid), act(),
                nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        else:  # inverted residual (depthwise)
            hid = cin * expand
            self.body = nn.Sequential(
                nn.Conv2d(cin, hid, 1, bias=False), nn.BatchNorm2d(hid), act(),
                nn.Conv2d(hid, hid, k, stride, p, groups=hid, bias=False), nn.BatchNorm2d(hid), act(),
                nn.Conv2d(hid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.se = SE(cout) if use_se else nn.Identity()
        self.short = None
        if {residual}:
            self.short = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.out_act = act()

    def forward(self, x):
        out = self.se(self.body(x))
        if self.short is not None:
            out = out + self.short(x)
        return self.out_act(out)


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        widths = {widths}
        depths = {depths}
        stem = {stem}
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem), act())
        layers, cin = [], stem
        for i, (w, d) in enumerate(zip(widths, depths)):
            for j in range(d):
                stride = 2 if (j == 0 and i > 0) else 1
                layers.append(Block(cin, w, stride, "{kind}", {k}, {expand}, {se}))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.drop = nn.Dropout({dropout})
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(self.drop(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)))
'''


def random_arch_code(rng):
    """Sample from a hand-designed space that is a superset of the seed families."""
    n_stages = rng.choice([2, 3, 3, 4])
    base = rng.choice([16, 24, 32, 48, 64])
    mult = rng.choice([1.5, 2.0])
    widths = [int(round(base * mult ** i / 8) * 8) for i in range(n_stages)]
    depths = [rng.choice([1, 2, 3]) for _ in range(n_stages)]
    p = dict(
        act=rng.choice(["ReLU", "GELU", "SiLU", "Hardswish"]),
        residual=rng.choice([True, True, False]),
        widths=widths, depths=depths, stem=rng.choice([16, 24, 32, 48]),
        kind=rng.choice(["basic", "bottleneck", "inverted"]), k=rng.choice([3, 3, 5]),
        expand=rng.choice([2, 3, 4, 6]), se=rng.choice([True, False]), dropout=rng.choice([0.0, 0.1, 0.2]),
    )
    idea = (f"random: {p['kind']} blocks, widths={widths}, depths={depths}, k={p['k']}, act={p['act']}, "
            f"residual={p['residual']}, se={p['se']}")
    return idea, _RANDOM_TEMPLATE.format(**p)
