"""Closed-loop rollout with the OpenVINO-compiled policy on Intel devices.

This is the Intel-deployment evidence: the exported IR runs entirely on
CPU (i9-13900K) and/or GPU (UHD 770) via OpenVINO -- no PyTorch in the loop.
The action-chunk queue mirrors select_action's n_action_steps behaviour.

Usage:
    python scripts/deploy_openvino.py --device CPU --task 0 --video
    python scripts/deploy_openvino.py --device GPU --task 1
    python scripts/deploy_openvino.py --device CPU --eps 10      # full eval
"""
import os, sys, json, argparse, collections
import numpy as np
import torch
import imageio.v3 as iio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.bimanual_env import BimanualSortEnv
from scripts.gen_demos import TASKS
from scripts.eval_closedloop import instruction_emb

CAM_KEYS = ["observation.images.overhead", "observation.images.left_wrist",
            "observation.images.right_wrist"]
N_STEPS = 20  # must match training n_action_steps


class OVPolicy:
    """Action-queue wrapper around a compiled OpenVINO model.
    The IR graph already contains normalize_inputs + unnormalize_outputs,
    so it consumes raw observations and emits raw joint targets."""

    def __init__(self, ir_path, device, n_steps=N_STEPS):
        import openvino as ov
        core = ov.Core()
        self.req = core.compile_model(core.read_model(ir_path), device)\
            .create_infer_request()
        self.n_steps = n_steps
        self.queue = collections.deque(maxlen=n_steps)

    def reset(self):
        self.queue.clear()

    def act(self, obs, emb_vec):
        if not self.queue:
            imgs = obs["images"]
            ins = {
                "state": obs["state"][None].astype(np.float32),
                "overhead": imgs["overhead_cam"].transpose(2, 0, 1)[None].astype(np.float32) / 255,
                "left_wrist": imgs["left_wrist_cam"].transpose(2, 0, 1)[None].astype(np.float32) / 255,
                "right_wrist": imgs["right_wrist_cam"].transpose(2, 0, 1)[None].astype(np.float32) / 255,
                "env_state": emb_vec[None].numpy().astype(np.float32),
            }
            out = self.req.infer(ins)
            actions = list(out.values())[0][0]      # (chunk, 12)
            self.queue.extend(actions[:self.n_steps])
        return np.asarray(self.queue.popleft(), dtype=np.float64)


def rollout(env, pol, emb_vec, max_steps=250, video=False):
    pol.reset()
    obs = env.obs()
    frames = []
    for _ in range(max_steps):
        a = pol.act(obs, emb_vec)
        obs = env.step(a)
        if video:
            frames.append(obs["images"]["overhead_cam"])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/act_lang")
    ap.add_argument("--ir", default="ckpt/openvino/model_fp16.xml")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    env = BimanualSortEnv()
    pol = OVPolicy(args.ir, args.device)
    print(f"running exported IR on OpenVINO device={args.device}")

    if args.task is not None or args.instruction is not None:
        tid = args.task if args.task is not None else 0
        text = args.instruction or TASKS[tid][2]
        emb = instruction_emb(text)
        env.grasp_assist = True
        env.assist_obj = TASKS[tid][0]
        env.reset(args.seed0, spawn_side=TASKS[tid][3],
                  spawn_obj=TASKS[tid][0])
        print(f"instruction: '{text}'")
        frames = rollout(env, pol, emb, args.max_steps, args.video)
        obj, bname = TASKS[tid][0], TASKS[tid][1]
        ok = env.in_bin(obj, bname)
        print(f"success={ok}")
        if args.video and frames:
            p = f"results/ov_rollout_{args.device.lower()}_t{tid}.mp4"
            iio.imwrite(p, np.stack(frames), fps=20)
            print(f"video -> {p}")
        return

    results = {}
    for tid, (obj, bname, text, _side) in TASKS.items():
        emb = instruction_emb(text)
        wins = 0
        for e in range(args.eps):
            env.grasp_assist = True
            env.assist_obj = obj
            env.reset(args.seed0 + 100 * tid + e,
                      spawn_side=TASKS[tid][3], spawn_obj=obj)
            rollout(env, pol, emb, args.max_steps)
            wins += env.in_bin(obj, bname)
        results[text] = f"{wins}/{args.eps}"
        print(f"task {tid}: '{text}' -> {wins}/{args.eps}")
    out = f"results/ov_eval_{args.device.lower()}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
