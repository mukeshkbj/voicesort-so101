# VoiceSort: pitch deck outline (export to PDF for submission)

## Slide 1: Title
VoiceSort: language-conditioned bimanual SO-101 manipulation, deployed on
Intel hardware via OpenVINO. AI Infra Summit Hackathon, Intel Online track.

## Slide 2: Problem
The Intel challenge: build a MuJoCo scene with SO-101 arms, collect LeRobot
demonstrations, train a VLA policy, and show it executing on Intel XPUs.

## Slide 3: Scene and system
- Custom dual-SO-101 MuJoCo scene (namespaced `<attach>` composition)
- 3 procedural objects, 2 bins, overhead + wrist cameras
- Voice/text instruction -> MiniLM semantic embedding -> ACT policy token
- Closed-loop 20 Hz control, 12-DoF position actuators

## Slide 4: Data
- 115 scripted LeRobot episodes (IK-scripted, physics-executed)
- Designated-arm task table: the instruction determines arm + object + bin
- Honest kinematic-carry data-gen shortcut, documented

## Slide 5: VLA
- ACT (52M) + `observation.environment_state` language slot
- Frozen MiniLM embeddings, so rephrased commands work (paraphrase 4/4)
- Instruction-swap video: the wrong instruction reroutes the correct arm

## Slide 6: Intel deployment
- ONNX -> OpenVINO IR (normalize + policy + denormalize all in-graph)
- i9-13900K CPU: 28.9 Hz policy rate, 3.2x PyTorch CPU
- UHD 770 iGPU: 28.6 Hz, real iGPU execution rather than a CPU fallback
- Closed-loop success on OV/CPU: 13/24 (matches torch 12/24 within noise)

## Slide 7: Voice layer
- Speechmatics STT -> keyword planner -> semantic embedding -> rollout
- Typed-command fallback; declines unparseable commands honestly

## Slide 8: Reproducibility and limits
- One-command data gen / train / eval / export / deploy scripts
- Apache-2.0 SO-101 assets + fully procedural objects (zero license risk)
- Limits: grasp-assist abstraction at eval, 3-task scope, ~50% success
  ceiling at 14k steps
