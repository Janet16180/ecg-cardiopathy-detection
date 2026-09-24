"""Experiment 010 CUDA profile with a numerical resume comparison.

The frozen runner's bitwise CUDA comparison rejected a 1.49e-8 difference
after save/restore. This entry point retains its profile and training recipe,
while checking the resumed model and AdamW states with a tight FP32 tolerance.
"""

import copy
import json

import torch

from scripts import run_cpc_crosslead as crosslead


ATOL = 1e-7
RTOL = 1e-5


def compare_state(left, right, path, maxima):
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Resume state keys differ at {path}")
        for key in left:
            compare_state(left[key], right[key], f"{path}.{key}", maxima)
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"Resume state sequence differs at {path}")
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            compare_state(item_left, item_right, f"{path}[{index}]", maxima)
    elif isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(f"Resume tensor missing at {path}")
        if left.is_floating_point():
            delta = float((left.cpu() - right.cpu()).abs().max())
            maxima[path.split(".", 1)[0]] = max(maxima.get(path.split(".", 1)[0], 0.0), delta)
            torch.testing.assert_close(left.cpu(), right.cpu(), atol=ATOL, rtol=RTOL, msg=path)
        else:
            torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0, msg=path)
    elif left != right:
        raise AssertionError(f"Resume scalar differs at {path}: {left!r} != {right!r}")


def profile_roundtrip(model, optimizer, signal, variant, generator):
    snapshot = {"model": crosslead.base.cpu_state(model),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "rng": crosslead.base.rng_state(generator)}
    duplicate = crosslead.CrossLeadPretrainer(variant).to(signal.device)
    duplicate.load_state_dict(snapshot["model"])
    copy_optimizer = torch.optim.AdamW(duplicate.parameters(), lr=1e-3, weight_decay=0.01)
    copy_optimizer.load_state_dict(snapshot["optimizer"])
    crosslead.base.restore_rng(snapshot["rng"], generator)
    crosslead.checked_step(model, optimizer, signal)
    expected_model = crosslead.base.cpu_state(model)
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    crosslead.base.restore_rng(snapshot["rng"], generator)
    crosslead.checked_step(duplicate, copy_optimizer, signal)
    maxima = {}
    compare_state(expected_model, crosslead.base.cpu_state(duplicate), "model", maxima)
    compare_state(expected_optimizer, copy_optimizer.state_dict(), "optimizer", maxima)
    crosslead.base.restore_rng(snapshot["rng"], generator)
    print(json.dumps({"stage": "profile_resume_check", "variant": variant,
                      "atol": ATOL, "rtol": RTOL, "max_abs_delta": maxima}), flush=True)
    return True


if __name__ == "__main__":
    crosslead.profile_roundtrip = profile_roundtrip
    crosslead.main()
