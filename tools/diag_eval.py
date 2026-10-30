"""Diagnostic evaluation, outside the study: evaluate a checkpoint under conditions closer to training collection.

The standard evaluation (``ndm_gym.evaluate --ckpt``) runs the policy's action as is, with clean observations. Training
collection adds Gaussian noise to the executed setpoint and clamps it to [-20, 20] (train.py), and noise to the
observations (``MDP.add_noise``). This script runs the unchanged evaluator with those parts switched on:
  DIAG_ACTION_NOISE=<std>  noise on the setpoint, then the clamp (as in collection)
  DIAG_CLAMP=1             the clamp alone
  DIAG_OBS_NOISE=1         the task's observation noise
  DIAG_SEED=<n>            seed of that noise (default 0)
Each result carries the settings under "diag". Usage, from the repository root, with the arguments of
``ndm_gym.evaluate --ckpt`` (keep --out_json outside logs/ and results/):
  DIAG_ACTION_NOISE=0.3 DIAG_OBS_NOISE=1 python tools/diag_eval.py --run_dir logs/<run> --ckpt <ckpt> --out_json <file>
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

import ndm_gym.evaluate as E  # noqa: E402

NOISE = float(os.environ.get("DIAG_ACTION_NOISE", "0"))
CLAMP = NOISE > 0 or os.environ.get("DIAG_CLAMP") == "1"
OBS = os.environ.get("DIAG_OBS_NOISE") == "1"
SEED = int(os.environ.get("DIAG_SEED", "0"))
_evaluate = E.evaluate


def evaluate(env, kin, motion, mdp, ck, record=None, push=None):
    torch.manual_seed(SEED)
    step, obs = env.step, mdp.obs
    if CLAMP:

        def step_as_collected(a):
            if NOISE:
                a = a + NOISE * torch.randn_like(a)
            return step(torch.clamp(a, -20, 20))

        env.step = step_as_collected
    if OBS:
        mdp.obs = lambda *args: mdp.add_noise(obs(*args))
    try:
        res = _evaluate(env, kin, motion, mdp, ck, record, push)
    finally:
        env.step, mdp.obs = step, obs
    res["diag"] = {"action_noise": NOISE, "clamp": CLAMP, "obs_noise": OBS, "seed": SEED}
    return res


E.evaluate = evaluate

if __name__ == "__main__":
    print(f"[diag] action noise {NOISE}, clamp {CLAMP}, observation noise {OBS}, seed {SEED}", flush=True)
    E.main()
