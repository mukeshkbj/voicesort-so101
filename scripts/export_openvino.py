"""Export the trained ACT policy to OpenVINO IR and benchmark on Intel devices.

The exported graph wraps normalize_inputs + ACT transformer (eval mode =
deterministic, zero VAE latent) + unnormalize_outputs, so it takes raw
observations and returns raw action chunks: exactly what the env needs.

Outputs:
    ckpt/openvino/model_fp32.xml|.bin
    ckpt/openvino/model_fp16.xml|.bin
    results/openvino_bench.json  (latency on CPU / GPU (UHD 770) / CUDA ref)
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_closedloop import load_policy, EMB_DIM

IMG = (3, 256, 256)
CAM_KEYS = ["observation.images.overhead", "observation.images.left_wrist",
            "observation.images.right_wrist"]


class ExportableACT(nn.Module):
    """normalize -> transformer -> unnormalize, flattened I/O for ONNX."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state, overhead, lwrist, rwrist, env_state):
        batch = {
            "observation.state": state,
            "observation.images.overhead": overhead,
            "observation.images.left_wrist": lwrist,
            "observation.images.right_wrist": rwrist,
            "observation.environment_state": env_state,
        }
        batch = self.policy.normalize_inputs(batch)
        batch["observation.images"] = [batch[k] for k in CAM_KEYS]
        actions, _ = self.policy.model(batch)      # (B, chunk, 12)
        return self.policy.unnormalize_outputs({"action": actions})["action"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/act_lang")
    ap.add_argument("--out", default="ckpt/openvino")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    policy, _ = load_policy(args.ckpt)
    policy.eval().cpu()
    wrapper = ExportableACT(policy).eval()

    dummies = (torch.randn(1, 12), torch.rand(1, *IMG), torch.rand(1, *IMG),
               torch.rand(1, *IMG), torch.randn(1, EMB_DIM))
    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "model.onnx")
    torch.onnx.export(wrapper, dummies, onnx_path, opset_version=17,
                      input_names=["state", "overhead", "left_wrist",
                                   "right_wrist", "env_state"],
                      output_names=["actions"])
    print(f"onnx -> {onnx_path}")

    import openvino as ov
    core = ov.Core()
    print("openvino devices:", core.available_devices)

    model = core.read_model(onnx_path)
    for tag, fp16 in [("fp32", False), ("fp16", True)]:
        p = os.path.join(args.out, f"model_{tag}")
        ov.save_model(model, p + ".xml", compress_to_fp16=fp16)
        print(f"ir -> {p}.xml")

    # ---- benchmark ----
    bench = {}
    inputs = {n: d.numpy() for n, d in zip(
        ["state", "overhead", "left_wrist", "right_wrist", "env_state"], dummies)}

    for dev in core.available_devices:
        try:
            cm = core.compile_model(model, dev)
            req = cm.create_infer_request()
            for _ in range(5):
                req.infer(inputs)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                req.infer(inputs)
            dt = (time.perf_counter() - t0) / args.iters
            bench[dev] = {"latency_ms": round(dt * 1e3, 2),
                          "policy_hz": round(1 / dt, 1)}
            print(f"{dev}: {dt*1e3:.2f} ms/policy-call ({1/dt:.0f} Hz)")
        except Exception as e:
            bench[dev] = {"error": str(e)}
            print(f"{dev}: FAILED - {e}")

    # PyTorch reference numbers for the report
    for tag, dev in [("torch_cpu", "cpu"),
                     ("torch_cuda", "cuda")]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        w = wrapper.to(dev)
        ins = tuple(d.to(dev) for d in dummies)
        with torch.no_grad():
            for _ in range(5):
                w(*ins)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                w(*ins)
            if dev == "cuda":
                torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
        bench[tag] = {"latency_ms": round(dt * 1e3, 2),
                      "policy_hz": round(1 / dt, 1)}
        print(f"{tag}: {dt*1e3:.2f} ms/policy-call ({1/dt:.0f} Hz)")

    os.makedirs("results", exist_ok=True)
    json.dump(bench, open("results/openvino_bench.json", "w"), indent=2)
    print(json.dumps(bench, indent=2))


if __name__ == "__main__":
    main()
