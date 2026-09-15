"""Smoke test: load bimanual scene, print structure, render a frame."""
import mujoco
import numpy as np

XML = "scene/bimanual_sort.xml"
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
print("joints:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)])
print("actuators:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])
print("cameras:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)])
print("sites:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)])

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)
img = renderer.render()
import imageio.v3 as iio
iio.imwrite("results/smoke_scene.png", img)
for cam in ["overhead_cam", "left_wrist_cam", "right_wrist_cam"]:
    try:
        renderer.update_scene(data, camera=cam)
        iio.imwrite(f"results/smoke_{cam}.png", renderer.render())
    except Exception as e:
        print(f"cam {cam}: {e}")
print("rendered results/smoke_scene.png")
