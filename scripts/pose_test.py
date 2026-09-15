"""Try candidate home poses for the dual arms; render a grid."""
import mujoco
import numpy as np
import imageio.v3 as iio

model = mujoco.MjModel.from_xml_path("scene/bimanual_sort.xml")
data = mujoco.MjData(model)

joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(12)]
print(joint_names)

poses = {
    "zero": np.zeros(12),
    "half_up": np.array([0, -0.8, 1.2, 0.4, 0, 0.8] * 2),
    "hover": np.array([0, -1.2, 1.5, 0.6, 0, 0.8] * 2),
    "deep": np.array([0, -1.4, 1.7, 0.8, 0, 0.8] * 2),
}

renderer = mujoco.Renderer(model, height=480, width=640)
for name, q in poses.items():
    data.qpos[:12] = q
    data.ctrl[:12] = q
    for _ in range(50):
        mujoco.mj_step(model, data)
    renderer.update_scene(data)
    iio.imwrite(f"results/pose_{name}.png", renderer.render())
    # print gripper positions
    for side in ["left", "right"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_gripperframe")
        print(name, side, np.round(data.site_xpos[sid], 3))
print("done")
