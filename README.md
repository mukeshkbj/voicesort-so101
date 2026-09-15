# VoiceSort — Language-Conditioned Bimanual SO-101 Manipulation in MuJoCo, deployed on Intel via OpenVINO

**AI Infra Summit Hackathon — Intel Online track submission.**

Two [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arms sit at a
table in a custom MuJoCo scene with three objects (red cube, blue sphere,
green cylinder) and two bins. Given a natural-language instruction — *"put the
red cube in the left bin"* — a language-conditioned **ACT** policy
(vision-language-action) reads the overhead + both wrist cameras and the
12-joint robot state, and drives both arms' position actuators closed-loop to
pick the correct object and drop it in the correct bin.

The policy is exported to **OpenVINO IR** and executed on Intel silicon:
**i9-13900K CPU** and **UHD 770 iGPU**. No IK, no scripting, no GPU vendor
lock-in at inference time.

## Hardware manifest

| Role | Device |
|------|--------|
| Training | NVIDIA RTX 4070 Ti (CUDA, PyTorch 2.6) |
| Deployment (required) | Intel Core i9-13900K CPU — OpenVINO |
| Deployment (accelerated) | Intel UHD 770 iGPU — OpenVINO |

## Results

### Closed-loop success (8 seeds × 3 tasks = 24 rollouts each)

| Runtime | red→left | blue→right | green→left | total |
|---|---|---|---|---|
| PyTorch (RTX 4070 Ti, fp32) | 6/8 | 3/8 | 3/8 | **12/24 (50%)** |
| OpenVINO FP32 — Intel CPU | 6/8 | 3/8 | 4/8 | **13/24 (54%)** |
| OpenVINO FP16 — Intel CPU | 5/8 | 1/8 | 2/8 | 8/24 (33%) |
| OpenVINO FP16 — UHD 770 iGPU | demo rollout: success | — | — | video: `results/ov_rollout_gpu.0_t1.mp4` |

Instruction-swap check (`results/swap_test.mp4`): spawning the task-0 layout
but instructing "blue sphere → right bin" sends the *right* arm after the
blue sphere — the language token, not the scene, selects the arm/object.

Paraphrase generalization: "place the red cube into the left bin please"
(cosine sim 0.983 to the canonical instruction) scores **4/4** — the MiniLM
conditioning accepts rephrased commands, not just the three training strings.

### Policy latency (50 calls, batch 1, full normalize→transformer→unnormalize graph)

| Runtime / device | Latency | Policy rate |
|---|---|---|
| OpenVINO — CPU (i9-13900K) | 34.7 ms | 28.9 Hz |
| OpenVINO — GPU.0 (UHD 770 iGPU) | 34.9 ms | 28.6 Hz |
| OpenVINO — GPU.1 (RTX 4070 Ti via OpenCL) | 36.8 ms | 27.2 Hz |
| PyTorch — CPU | 109.6 ms | 9.1 Hz |
| PyTorch — CUDA (RTX 4070 Ti, ref) | 18.2 ms | 55.0 Hz |

**OpenVINO on the i9-13900K CPU runs the full VLA policy 3.2× faster than
PyTorch CPU (28.9 vs 9.1 Hz)** and the UHD 770 iGPU matches it at 28.6 Hz —
both well above the 20 Hz control loop.

## Pipeline

```
scripts/gen_demos.py      scripted IK demos -> LeRobot dataset (task text kept)
scripts/train_act_lang.py ACT + MiniLM language token -> ckpt/act_lang
scripts/eval_closedloop.py  torch closed-loop eval + rollout videos
scripts/export_openvino.py  ONNX -> OpenVINO IR (fp32/fp16) + device benchmark
scripts/deploy_openvino.py  closed-loop rollout driven by OpenVINO runtime
```

### Language conditioning (the "L" in VLA)

Each LeRobot episode stores its instruction string. At training time the
string is embedded once with frozen `all-MiniLM-L6-v2` (384-d, semantic) and
fed to ACT through its `observation.environment_state` input — an encoder
token (`FeatureType.ENV`, identity normalization). At inference the same
embedding is computed from any instruction text, so the policy responds to
rephrased commands, not just the three training strings. Ablate/verify with
the instruction-swap test in `eval_closedloop.py`.

### Data generation honesty notes

- `env/bimanual_env.py::grasp_obj` uses a kinematic carry shortcut (object
  attaches to the gripper once the closed gripper is within 5 cm) — a standard
  scripted-demonstration trick used inside the data generator.
- Eval/deploy rollouts use the matching `grasp_assist` abstraction: the
  *policy* chooses when/where to close and open the gripper; the assist only
  decides whether a close within 9 cm of the target object counts as a
  successful pick. Arm control is 100% the learned policy; no IK runs at
  inference. Finger-level contact grasping is a documented limitation.
- Each task has a designated arm and its target object always spawns on that
  arm's side, so arm selection is determined by the instruction embedding —
  this mirrors a "designated worker" bimanual cell and removes a
  vision-based arm-routing ambiguity the language token could not resolve.

## Reproduce

```bash
# python 3.10 env
pip install -r requirements.txt

# 1. scene smoke test
python scripts/smoke_scene.py

# 2. dataset (~18 min, 120 attempts -> ~105 kept)
python scripts/gen_demos.py 120

# 3. train on CUDA GPU (~2-3h for 15-20k steps, batch 32)
python scripts/train_act_lang.py --steps 20000 --batch 32 --out ckpt/act_lang
# resume after interruption:
python scripts/train_act_lang.py --steps 10000 --out ckpt/act_lang_r2 \
    --resume ckpt/act_lang_step5000

# 4. closed-loop eval (torch)
python scripts/eval_closedloop.py --ckpt ckpt/act_lang --eps 10

# 5. export + benchmark on Intel devices
python scripts/export_openvino.py --ckpt ckpt/act_lang --out ckpt/openvino

# 6. closed-loop rollout executed by OpenVINO on Intel CPU / iGPU
python scripts/deploy_openvino.py --device CPU --task 0 --video
python scripts/deploy_openvino.py --device GPU.0 --task 1 --video

# 7. command-line front end (typed or Speechmatics voice)
python scripts/voice_demo.py --text "put the green cylinder in the left bin" --video
SPEECHMATICS_API_KEY=... python scripts/voice_demo.py --wav command.wav --video
```

## Repo layout

- `scene/` — MuJoCo MJCF: `bimanual_sort.xml` (dual-arm scene), `so101_arm.xml`
  (arm model with wrist camera), `assets/` (SO-101 STLs, Apache-2.0)
- `env/bimanual_env.py` — env wrapper: obs/action spec, seeded reset, success
  check, scratch-data IK (data-gen only), kinematic carry (data-gen only)
- `policy/text_embed.py` — MiniLM instruction embeddings
- `scripts/` — pipeline scripts above
- `results/` — logs, benchmark JSONs, rollout videos
- `demo_data/` — generated LeRobot dataset (not committed)
- `ckpt/` — checkpoints + OpenVINO IR (not committed; see release notes)

## Training notes

The shipped checkpoint is `act_v2_step10000` — ~14k cumulative optimizer
steps (batch 32, lr 1e-4, chunk 50 / n_action_steps 20, ~2.5 h on the 4070 Ti
including resume chunks). Success rate oscillates between checkpoints on
this small dataset; `step10000` was selected by held-out-seed eval.

## Licenses

SO-101 MJCF/STL: Apache-2.0 (TheRobotStudio/SO-ARM100). All other scene
geometry is procedural MuJoCo primitives. Dependency licenses in
[LICENSES.md](LICENSES.md). Project code: MIT.

## Known limitations

- Scripted demos use kinematic carry (see note above); the learned policy
  must still physically track/act through contacts, which is the dominant
  source of eval failures.
- Task distribution is small (3 instructions); the language token is semantic
  but generalization is bounded by the task set.
