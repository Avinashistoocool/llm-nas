import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        set_by_path(cfg, key, value)
    root = cfg["task"]["data_root"]
    if not os.path.isabs(root):
        cfg["task"]["data_root"] = os.path.join(ROOT, root)
    return cfg


def set_by_path(cfg, dotted, value):
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_override(text):
    """'search.budget=50' -> ('search.budget', 50)"""
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)


def clone(cfg):
    return copy.deepcopy(cfg)
