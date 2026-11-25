"""RelayVSR 权重格式识别与元数据校验。

官方权重的 safetensors metadata 中带有 ``format`` 与 ``config`` 字段：
  x2：relayvsr-wan-x2-v1 / relayvsr-flash-x2-v1
  x4：refvsr-wan-v1     / refvsr-flash-v1
插件的加载路径完全依据这些字段，而不是文件名。
"""
# Portions of this file are derived from RelayVSR (https://github.com/kopperx/RelayVSR)
# and LightX2V (https://github.com/ModelTC/LightX2V), both under the Apache License 2.0.
# Modifications: adapted for this ComfyUI plugin.

import json
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

WAN_FORMATS = {"relayvsr-wan-x2-v1": 2, "refvsr-wan-v1": 4}
FLASH_FORMATS = {"relayvsr-flash-x2-v1": 2, "refvsr-flash-v1": 4}

FLASH_DEFAULTS = {
    "condition_channels": 48,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "dim": 512,
    "depth": 10,
    "num_heads": 8,
    "num_kv_groups": 2,
}


@dataclass
class BundleInfo:
    directory: Path
    wan_path: Path
    flash_path: Path
    scale: int

    @property
    def label(self):
        return f"{self.directory.name} (x{self.scale})"


def read_metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def inspect_safetensors(path):
    """返回 (component, scale)；非 RelayVSR 权重返回 None。"""
    try:
        metadata = read_metadata(path)
    except Exception:
        return None
    fmt = metadata.get("format")
    if fmt in WAN_FORMATS:
        return "wan", WAN_FORMATS[fmt]
    if fmt in FLASH_FORMATS:
        return "flash", FLASH_FORMATS[fmt]
    return None


def inspect_bundle(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return None
    wan_path = flash_path = None
    scale = None
    for path in sorted(directory.glob("*.safetensors")):
        info = inspect_safetensors(path)
        if info is None:
            continue
        component, component_scale = info
        if scale is None:
            scale = component_scale
        elif scale != component_scale:
            raise ValueError(f"Directory mixes weights of different scales: {directory}")
        if component == "wan":
            wan_path = path
        else:
            flash_path = path
    if wan_path is None or flash_path is None:
        return None
    return BundleInfo(directory=directory, wan_path=wan_path, flash_path=flash_path, scale=scale)


def checkpoint_config(path, component, scale):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        expected = (
            f"relayvsr-{component}-x2-v1" if scale == 2 else f"refvsr-{component}-v1"
        )
        if metadata.get("format") != expected:
            raise ValueError(
                f"Weight format mismatch: expected {expected}, got {metadata.get('format')!r} ({path})"
            )
        config = json.loads(metadata["config"])
        if component == "wan":
            # 官方 x4 Wan metadata 早于显式 scale 字段。
            if scale == 4:
                config.setdefault("upscale_factor", 4)
            unshuffle = 8 if scale == 2 else 16
            hidden = config.get("lq_proj_hidden_dim1")
            expected_shape = [hidden, 3 * unshuffle**2, 1, 1]
            actual_shape = list(handle.get_slice("lq_proj.conv1.weight").get_shape())
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Wan input projection does not match native x{scale} geometry: "
                    f"{actual_shape} != {expected_shape} ({path})"
                )
        else:
            config = {**FLASH_DEFAULTS, **config}
    return config


def validate_configs(wan_config, flash_config, scale):
    if list(wan_config.get("patch_size", [1, 2, 2])) != [1, 2, 2]:
        raise ValueError("RelayVSR only supports Wan backbones with patch_size=[1,2,2]")
    if wan_config.get("in_channels") not in (None, 48) or wan_config.get("out_channels") not in (None, 48):
        raise ValueError("RelayVSR Wan backbone must have 48 input/output channels")
    if wan_config.get("upscale_factor") not in (None, scale):
        raise ValueError("Wan weight native scale does not match the bundle directory")
    if flash_config.get("upscale_factor") not in (None, scale):
        raise ValueError("Flash weight native scale does not match the bundle directory")
    if flash_config.get("variant") not in (None, "S"):
        raise ValueError("RelayVSR only supports Flash variant S")
    if flash_config.get("condition_channels") not in (None, 48):
        raise ValueError("RelayVSR Flash model must take 48-channel condition input")
