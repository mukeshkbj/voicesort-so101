"""Hand-rolled damped least-squares IK on gripperframe site -> verify reach."""
import mujoco
import numpy as np
import imageio.v3 as iio

model = mujoco.MjModel.from_xml_path("scene/bimanual_sort.xml")
data = mujoco.MjData(model)

HOME = np.array([0, -1.0, 1.3, 0.5, 0, 0.8] * 2)
data.qpos[:12] = HOME
data.ctrl[:12] = HOME
mujoco.mj_forward(model, data)

JOINTS = {s: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{j}")
              for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]]
          for s in ["left", "right"]}
# qpos address of each joint
QADR = {s: [model.jnt_qposadr[j] for j in ids] for s, ids in JOINTS.items()}
SITES = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_gripperframe") for s in JOINTS}
LO = model.jnt_range[:, 0]
HI = model.jnt_range[:, 1]


def dls_ik(side, target, iters=2000, alpha=0.6, lam=0.08, tol=2e-3):
    """Damped least-squares position IK on the gripperframe site. Writes qpos."""
    site = SITES[side]
    qadr = np.array(QADR[side])
    for i in range(iters):
        mujoco.mj_forward(model, data)
        err = np.asarray(target) - data.site_xpos[site]
        if np.linalg.norm(err) < tol:
            return True, i
        J = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, J, None, site)
        Ja = J[:, model.jnt_dofadr[JOINTS[side]]]
        dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + lam**2 * np.eye(3), err)
        q = data.qpos[qadr] + alpha * dq
        data.qpos[qadr] = np.clip(q, LO[qadr], HI[qadr])
    return False, iters


red = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_red_cube")
green = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_green_cyl")
binl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin_left")
binr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin_right")

for name, tgt in [
    ("left->hover", np.array([-0.10, 0.05, 0.15])),
    ("left->red", data.xpos[red] + np.array([0, 0, 0.015])),
    ("left->lift", data.xpos[red] + np.array([0, 0, 0.12])),
    ("left->bin", data.xpos[binl] + np.array([0, 0, 0.12])),
]:
    ok, it = dls_ik("left", tgt)
    print(f"{name}: ok={ok} iters={it} site={np.round(data.site_xpos[SITES['left']],3)} tgt={np.round(tgt,3)}")

for name, tgt in [
    ("right->hover", np.array([0.10, 0.05, 0.15])),
    ("right->green", data.xpos[green] + np.array([0, 0, 0.015])),
    ("right->bin", data.xpos[binr] + np.array([0, 0, 0.12])),
]:
    ok, it = dls_ik("right", tgt)
    print(f"{name}: ok={ok} iters={it} site={np.round(data.site_xpos[SITES['right']],3)}")

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="overhead_cam")
iio.imwrite("results/ik_test_overhead.png", renderer.render())
renderer.update_scene(data)
iio.imwrite("results/ik_test.png", renderer.render())
print("saved")
