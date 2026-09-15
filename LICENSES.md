# Asset & Dependency License Manifest

## Simulation assets

| Asset | Source | License | Notes |
|-------|--------|---------|-------|
| SO-101 arm MJCF + STL meshes (`scene/so101_arm.xml`, `scene/assets/`) | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Apache-2.0 | Public robot model published by the arm's manufacturer for simulation use |
| Tabletop, bins, primitive objects (cube/sphere/cylinder) | generated in `scene/bimanual_sort.xml` | n/a | Procedural MJCF primitives authored for this project — no external 3D assets, so no third-party licensing risk |

No Sketchfab/CGTrader/Free3D assets were used; all scene geometry beyond the
SO-101 arm is procedural MuJoCo primitives, which avoids license issues
entirely.

## Software dependencies

| Package | License |
|---------|---------|
| MuJoCo | Apache-2.0 |
| PyTorch / torchvision | BSD-3 |
| Hugging Face LeRobot | Apache-2.0 |
| Hugging Face transformers | Apache-2.0 |
| sentence-transformers + all-MiniLM-L6-v2 | Apache-2.0 |
| Intel OpenVINO | Apache-2.0 |
| ONNX / onnxscript | MIT / Apache-2.0 |
| imageio, av (PyAV), einops, safetensors, numpy | BSD/MIT |

## Reference code

- `ref-lerobot-mujoco-tutorial/` (jeongeun980906) — used as a reference for the
  LeRobot dataset/training API; MIT license. Not vendored into the project.
- `ref-SO-ARM100/` — reference clone of the upstream asset repo; Apache-2.0.
