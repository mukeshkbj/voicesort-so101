# Lablab.ai submission copy — paste-ready

## Project title
VoiceSort: language-conditioned bimanual SO-101 sorting, deployed on Intel via OpenVINO

## Repository
https://github.com/mukeshkbj/voicesort-so101

## Short description (tagline field)
Two SO-101 arms sort objects by spoken instruction. A MiniLM-conditioned ACT
policy runs closed-loop in MuJoCo through OpenVINO on an i9-13900K CPU and
UHD 770 iGPU.

## Long description

We built the Intel Online track challenge end to end: a custom MuJoCo scene
with two SO-101 arms (public Apache-2.0 assets, procedural objects so there
is zero licensing risk), a scripted-demonstration pipeline that writes
LeRobot-format data with the natural-language task string kept per episode,
and a vision-language-action policy that runs the robot.

The "L" is real: each instruction is embedded once with a frozen MiniLM
encoder and fed to ACT as a transformer token. Because the conditioning is
semantic, the policy follows rephrased commands, not just the three training
strings. An instruction-swap rollout shows the language token selecting which
arm moves and which object it goes for.

Deployment is the required part and it is measured, not claimed: the policy
is exported through ONNX to OpenVINO IR with normalization baked into the
graph, then executed closed-loop on the i9-13900K CPU (28.9 Hz policy rate)
and on the UHD 770 iGPU (28.6 Hz). OpenVINO on the CPU alone is 3.2x faster
than PyTorch on the same chip. Success rate on held-out seeds: 13/24 with the
FP32 IR, versus 12/24 for the PyTorch reference.

A voice layer (Speechmatics STT -> keyword planner -> the same semantic
embedding) drives the whole thing from a spoken command; the video shows a
recorded instruction going end to end on OpenVINO/CPU.

## Demo video / presentation
- Pitch deck: `deck/pitch.pdf` in the repo
- OpenVINO/CPU rollout: `results/ov_rollout_cpu_t0.mp4`
- OpenVINO/UHD-770 iGPU rollout: `results/ov_rollout_gpu.0_t1.mp4`
- Voice command end-to-end: `results/voice_demo.mp4`
- Instruction-swap: `results/swap_test.mp4`

## Technologies used
MuJoCo, LeRobot, PyTorch, OpenVINO, ONNX, Hugging Face transformers,
sentence-transformers (all-MiniLM-L6-v2), Speechmatics, Python

## Hardware
Trained on RTX 4070 Ti. Deployed and benchmarked on Intel i9-13900K CPU and
Intel UHD 770 iGPU (both are in-scope Intel XPUs per organizer guidance).

## Team
mukeshkbj
