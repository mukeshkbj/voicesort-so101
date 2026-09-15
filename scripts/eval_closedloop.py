"""Closed-loop evaluation of the language-conditioned ACT policy in MuJoCo.

Runs N seeded rollouts per instruction, policy drives all 12 joints through
the position actuators -- no IK, no scripting at inference time. Records
overhead-cam videos and reports per-task success (object in target bin).

Usage:
    python scripts/eval_closedloop.py --ckpt ckpt/act_lang --eps 10
    python scripts/eval_closedloop.py --ckpt ckpt/act_lang --eps 5 --instruction "put the red cube in the left bin" --video
"""
import os, sys, json, argparse
import numpy as np
import torch
import imageio.v3 as iio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.bimanual_env import BimanualSortEnv
from scripts.gen_demos import TASKS
from policy.text_embed import embed_texts

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

EMB_DIM = 384


def load_policy(ckpt, data="demo_data", repo_id="voicesort_so101"):
    """Load a saved checkpoint: config.json restores features/chunk size and
    the safetensors buffers restore the normalization stats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy.from_pretrained(ckpt)
    policy.eval().to(device)
    return policy, device


def instruction_emb(text):
    """Free-form instruction -> semantic embedding (MiniLM)."""
    return torch.from_numpy(embed_texts([text]))[0]


def rollout(env, policy, device, emb_vec, max_steps=400, video=False):
    """One closed-loop episode: policy picks an action every control step.
    select_action internally queues chunks of n_action_steps and refills as
    needed; call policy.reset() before each episode to clear the queue."""
    policy.reset()
    obs = env.obs()
    frames = []
    for t in range(max_steps):
        imgs = obs["images"]
        batch = {
            "observation.state": torch.from_numpy(obs["state"]).unsqueeze(0),
            "observation.images.overhead": torch.from_numpy(imgs["overhead_cam"]).permute(2, 0, 1).unsqueeze(0).float() / 255,
            "observation.images.left_wrist": torch.from_numpy(imgs["left_wrist_cam"]).permute(2, 0, 1).unsqueeze(0).float() / 255,
            "observation.images.right_wrist": torch.from_numpy(imgs["right_wrist_cam"]).permute(2, 0, 1).unsqueeze(0).float() / 255,
            "observation.environment_state": emb_vec.unsqueeze(0),
        }
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            action = policy.select_action(batch)  # (B, 12)
        obs = env.step(action[0].cpu().numpy())
        if video:
            frames.append(obs["images"]["overhead_cam"])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/act_lang")
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=250)  # env control steps
    args = ap.parse_args()

    env = BimanualSortEnv()
    policy, device = load_policy(args.ckpt)

    if args.instruction or args.task is not None:
        tid = args.task if args.task is not None else 0
        text = args.instruction or TASKS[tid][2]
        emb_vec = instruction_emb(text)
        env.grasp_assist = True
        env.assist_obj = TASKS[tid][0]
        env.reset(args.seed0, spawn_side=TASKS[tid][3],
                  spawn_obj=TASKS[tid][0])
        print(f"instruction: '{text}'")
        frames = rollout(env, policy, device, emb_vec,
                         max_steps=args.max_steps, video=args.video)
        obj, bname = TASKS[tid][0], TASKS[tid][1]
        ok = env.in_bin(obj, bname)
        print(f"success={ok}")
        if args.video and frames:
            iio.imwrite(f"results/rollout_{tid}.mp4", np.stack(frames), fps=20)
            print(f"video -> results/rollout_{tid}.mp4")
        return

    # full eval: all tasks x eps seeds
    results = {}
    for tid, (obj, bname, text, _side) in TASKS.items():
        emb_vec = instruction_emb(text)
        wins = 0
        for e in range(args.eps):
            env.grasp_assist = True
            env.assist_obj = obj
            env.reset(args.seed0 + 100 * tid + e,
                      spawn_side=TASKS[tid][3], spawn_obj=obj)
            rollout(env, policy, device, emb_vec, max_steps=args.max_steps)
            ok = env.in_bin(obj, bname)
            wins += ok
        results[text] = f"{wins}/{args.eps}"
        print(f"task {tid}: '{text}' -> {wins}/{args.eps}")
    json.dump(results, open("results/eval_closedloop.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
