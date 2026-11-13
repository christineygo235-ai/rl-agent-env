"""Policy server: runs INSIDE the sandbox, next to untrusted agent code.

    python -E -s -B policy_server.py --policy-dir runs/r1/policy     (cwd = workspace)

Loads the agent's `policy.py` (`load(path)` -> object with `act(obs)` and
optionally `reset()`), then answers requests over a line-delimited JSON
protocol on the ORIGINAL stdout file descriptor:

    <- {"ready": true}                      | {"error": "..."}   (after load)
    -> {"op": "reset"}                      <- {"ok": true}
    -> {"op": "act", "obs": <encoded>}      <- {"action": <encoded>} | {"error": "..."}
    -> {"op": "close"}

Anything the agent prints goes to stderr: fd 1 is re-pointed at fd 2 after
the protocol fd has been duplicated, so prints cannot corrupt or forge
protocol messages. This file is deliberately standalone (stdlib + numpy, no
rlenv imports). The harness trusts nothing it says beyond a well-formed
action, which it validates against the action space itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def decode_obs(enc):
    import numpy as np

    if isinstance(enc, dict) and enc.get("t") == "a":
        return np.asarray(enc["v"], dtype=enc["dtype"]).reshape(enc["shape"])
    return enc


def encode_action(action):
    import numpy as np

    if isinstance(action, np.ndarray):
        return {"t": "a", "v": action.tolist(), "dtype": str(action.dtype), "shape": list(action.shape)}
    if isinstance(action, np.generic):
        return action.item()
    if isinstance(action, (bool, int, float)):
        return action
    raise TypeError(f"act() returned unsupported type {type(action).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", required=True)
    args = ap.parse_args()

    proto = os.fdopen(os.dup(1), "w", buffering=1)  # protocol channel
    os.dup2(2, 1)                                   # agent prints now go to stderr
    sys.stdout = sys.stderr

    def send(msg: dict) -> None:
        proto.write(json.dumps(msg) + "\n")
        proto.flush()

    # Import agent modules from the workspace (cwd), not from this file's directory.
    sys.path[0] = os.getcwd()
    try:
        import policy as agent_policy  # noqa: PLC0415

        pol = agent_policy.load(args.policy_dir)
        if not callable(getattr(pol, "act", None)):
            raise TypeError("load() must return an object with an act(obs) method")
    except BaseException as exc:  # report, then exit; the harness decides what it means
        send({"error": f"policy load failed: {type(exc).__name__}: {exc}"})
        traceback.print_exc()
        return
    send({"ready": True})

    for line in sys.stdin:
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "act":
                send({"action": encode_action(pol.act(decode_obs(req["obs"])))})
            elif op == "reset":
                reset = getattr(pol, "reset", None)
                if callable(reset):
                    reset()
                send({"ok": True})
            elif op == "close":
                return
            else:
                send({"error": f"unknown op {op!r}"})
        except BaseException as exc:
            send({"error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()


if __name__ == "__main__":
    main()
