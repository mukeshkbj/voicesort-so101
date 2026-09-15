"""Train language-conditioned ACT on the VoiceSort bimanual dataset.

Language conditioning is injected through ACT's unused `observation.
environment_state` slot: each frame's task_index maps to a frozen MiniLM
embedding of its instruction string, so the transformer encoder sees a
language token. Text is only needed at data-collection/inference time via the
task table; the policy itself runs on a fixed-size vector.
"""
import os, sys, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.text_embed import task_embeddings, EMB_DIM

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset, LeRobotDatasetMetadata)
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.common.datasets.factory import resolve_delta_timestamps


class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, x):
        return (torch.randn(x.size()) * self.std + x).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--repo_id", default="voicesort_so101")
    ap.add_argument("--out", default="ckpt/act_lang")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--n_action_steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir to resume weights from")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--step_offset", type=int, default=0,
                    help="added to step for checkpoint naming (resumed runs)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.data)
    feats = dataset_to_policy_features(meta.features)
    out_feats = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    in_feats = {k: f for k, f in feats.items() if k not in out_feats}
    # dataset features store HWC; policy features must be CHW
    in_feats = {k: (PolicyFeature(f.type, (f.shape[2], f.shape[0], f.shape[1]))
                  if f.type is FeatureType.VISUAL else f)
                for k, f in in_feats.items()}
    # language token slot
    in_feats["observation.environment_state"] = PolicyFeature(
        FeatureType.ENV, (EMB_DIM,))

    cfg = ACTConfig(input_features=in_feats, output_features=out_feats,
                    chunk_size=args.chunk, n_action_steps=args.n_action_steps)
    delta_ts = resolve_delta_timestamps(cfg, meta)

    # stats: identity normalization for the language vector
    stats = dict(meta.stats)
    stats["observation.environment_state"] = {
        "mean": torch.zeros(EMB_DIM), "std": torch.ones(EMB_DIM),
        "min": torch.full((EMB_DIM,), -1.0), "max": torch.ones(EMB_DIM),
    }

    # frozen instruction embeddings keyed by task_index
    task_list = [meta.tasks[i] for i in sorted(meta.tasks.keys())]
    emb = torch.from_numpy(task_embeddings(task_list))  # (n_tasks, 384)
    print(f"tasks: {task_list}")
    np.save(os.path.join("results", "task_embeddings.npy"), emb.numpy())
    with open(os.path.join("results", "task_list.json"), "w") as f:
        import json; json.dump(task_list, f)

    if args.resume:
        policy = ACTPolicy.from_pretrained(args.resume)
        print(f"resumed weights from {args.resume}")
    else:
        policy = ACTPolicy(cfg, dataset_stats=stats)
    policy.train().to(device)

    ds = LeRobotDataset(args.repo_id, root=args.data, delta_timestamps=delta_ts,
                        image_transforms=transforms.Compose([
                            AddGaussianNoise(0.02),
                            transforms.Lambda(lambda x: x.clamp(0, 1))]))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    pin_memory=device.type == "cuda", drop_last=True)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    t0 = time.time()
    step = 0
    done = False
    while not done:
        for batch in dl:
            if step >= args.steps:
                done = True
                break
            batch["observation.environment_state"] = \
                emb[batch["task_index"].long()]
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)
            loss, info = policy.forward(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 200 == 0:
                print(f"step {step} loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s) {info}", flush=True)
            if step % args.save_every == 0 and step > 0:
                ck = f"{args.out}_step{step + args.step_offset}"
                policy.save_pretrained(ck)
                print(f"checkpoint -> {ck}", flush=True)
            step += 1

    policy.save_pretrained(args.out)
    print(f"saved checkpoint -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
