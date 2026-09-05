import torch.nn as nn
from dataclasses import dataclass
from typing import Iterator, Any
import re


def fullmatch_shell(pattern, string):
    regex = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.fullmatch(regex, string) is not None


@dataclass
class ParamCfg:
    """
    ParamCfg to use with `opt_param_cfg()`.

    Attributes:
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer. If None, use the default value of the optimizer.
    """

    lr: float | None = None
    weight_decay: float | None = None
    param: nn.Parameter | Iterator[nn.Parameter] | None = None


def opt_param_cfg(
    parameters: Iterator[tuple[str, nn.Parameter]], config: dict[str, ParamCfg], other_params: list[tuple[nn.Parameter | Iterator[nn.Parameter], ParamCfg]] = []
) -> Iterator[dict[str, Any]]:
    """
    Apply parameter-specific configurations to model parameters for optimizer setup.

    Matches parameter names against patterns in the config dictionary and yields
    configuration dictionaries for parameters that match, along with unmatched parameters.

    Args:
        parameters: Iterator of (name, parameter) tuples from the model.
        config: Dictionary mapping name patterns to ParamCfg objects.
        other_params: List of ParamCfg objects for parameters not matched by the config.

    Yields:
        Configuration dictionaries for matched parameters, or (name, param) tuples for unmatched parameters.
    """

    keys = {k for k in config.keys()}
    for name, param in parameters:
        out: dict[str, Any] = {"params": param}
        for pattern, cfg in config.items():
            if fullmatch_shell(pattern, name):
                if pattern in keys:
                    keys.remove(pattern)
                param_cfg = cfg
                if param_cfg.lr is not None:
                    out["lr"] = param_cfg.lr
                if param_cfg.weight_decay is not None:
                    out["weight_decay"] = param_cfg.weight_decay
                break
        yield out
    assert not keys, f"Unmatched config keys: {keys}"

    for param, cfg in other_params:
        out = {"params": param}
        if cfg.lr is not None:
            out["lr"] = cfg.lr
        if cfg.weight_decay is not None:
            out["weight_decay"] = cfg.weight_decay
        yield out
