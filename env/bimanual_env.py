"""Bimanual SO-101 sorting environment.

Wraps the MuJoCo scene: state/actions are 12-dim (2 arms x 6 joints incl.
grippers), observations are overhead + both wrist cameras. IK runs on a
scratch MjData so planning never perturbs the live sim; execution is via the
position actuators (real physics, real contacts).
"""
import mujoco
import numpy as np

XML = "scene/bimanual_sort.xml"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper"]
SIDES = ["left", "right"]
OBJECTS = ["obj_red_cube", "obj_blue_sphere", "obj_green_cyl"]
BINS = ["bin_left", "bin_right"]

HOME = np.array([0.0, -1.0, 1.3, 0.5, 0.0, 0.8] * 2, dtype=np.float64)
GRIP_OPEN = 0.8
GRIP_CLOSED = -0.1
# spawn zone on the table in front of the arms
SPAWN_X = (-0.12, 0.12)
SPAWN_Y = (0.03, 0.13)
OBJ_Z = {"obj_red_cube": 0.018, "obj_blue_sphere": 0.02, "obj_green_cyl": 0.022}


class BimanualSortEnv:
    def __init__(self, xml=XML, img_size=256, control_hz=20):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        # scratch data for IK planning (never disturbs live state)
        self._plan = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=img_size, width=img_size)
        self.img_size = img_size
        self.substeps = int(round(1.0 / (control_hz * self.model.opt.timestep)))
        self.control_dt = 1.0 / control_hz

        self.qadr = {s: np.array([self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{j}")]
            for j in ARM_JOINTS]) for s in SIDES}
        self.dadr = {s: np.array([self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{j}")]
            for j in ARM_JOINTS]) for s in SIDES}
        self.act_id = {s: np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_{j}")
            for j in ARM_JOINTS]) for s in SIDES}
        self.site = {s: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_gripperframe") for s in SIDES}
        self.obj_body = {o: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, o) for o in OBJECTS}
        self.obj_jadr = {o: self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{o}_joint")]
            for o in OBJECTS}
        self.bin_body = {b: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in BINS}
        self.joint_lo = self.model.jnt_range[:, 0]
        self.joint_hi = self.model.jnt_range[:, 1]
        self.rng = np.random.default_rng(0)
        self.carry_obj = None      # object name kinematically held, or None
        self.carry_off = np.zeros(3)
        # grasp-assist mode for POLICY rollouts: reproduces the scripted-demo
        # carry semantics (gripper-close near object picks it up, opening
        # drops it). Still the learned policy that controls the arms.
        self.grasp_assist = False
        self.assist_obj = None     # which object the assist may pick up
        self.assist_thresh = 0.06
        self.reset(0)

    # ---------- state ----------
    def reset(self, seed=None, spawn_side=None, spawn_obj=None):
        """spawn_side='left'|'right' + spawn_obj forces that object into the
        matching x half-range so its designated arm always serves it."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.carry_obj = None
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:12] = HOME
        self.data.ctrl[:12] = HOME
        half = {"left": (-0.12, -0.02), "right": (0.02, 0.12)}
        # scatter objects without overlap
        placed = []
        for o in OBJECTS:
            xr = half[spawn_side] if (o == spawn_obj and spawn_side) else SPAWN_X
            for _ in range(100):
                x = self.rng.uniform(*xr)
                y = self.rng.uniform(*SPAWN_Y)
                if all((x - px) ** 2 + (y - py) ** 2 > 0.05 ** 2 for px, py in placed):
                    break
            placed.append((x, y))
            adr = self.obj_jadr[o]
            self.data.qpos[adr:adr + 3] = [x, y, OBJ_Z[o]]
            self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        return self.obs()

    def obs(self):
        mujoco.mj_forward(self.model, self.data)
        imgs = {}
        for cam in ["overhead_cam", "left_wrist_cam", "right_wrist_cam"]:
            self.renderer.update_scene(self.data, camera=cam)
            imgs[cam] = self.renderer.render().copy()
        state = np.concatenate([self.data.qpos[a] for a in
                                [self.qadr["left"], self.qadr["right"]]]).astype(np.float32)
        return {"images": imgs, "state": state}

    def step(self, ctrl14):
        self.data.ctrl[:12] = ctrl14
        if self.grasp_assist:
            self._assist_update(ctrl14)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if self.carry_obj is not None:
                self._carry_update()
        return self.obs()

    def _assist_update(self, ctrl):
        """Mirror scripted-grasp semantics for learned-policy rollouts.
        Gripper channel <0.3 = 'closed', >0.5 = 'open' (same thresholds as
        the demos' GRIP_CLOSED=-0.1 / GRIP_OPEN=0.8 endpoints)."""
        for i, side in enumerate(SIDES):
            grip_cmd = ctrl[self.act_id[side][5]]
            if self.carry_obj is None and self.assist_obj is not None \
                    and grip_cmd < 0.3:
                if np.linalg.norm(self.site_pos(side)
                                  - self.obj_pos(self.assist_obj)) \
                        < self.assist_thresh:
                    self.carry_obj = self.assist_obj
                    self.carry_side = side
                    self.carry_off = (self.obj_pos(self.assist_obj)
                                      - self.site_pos(side))
            elif self.carry_obj is not None and self.carry_side == side \
                    and grip_cmd > 0.5:
                self.carry_obj = None

    # ---------- scripted grasp assist (data-gen only) ----------
    def grasp_obj(self, side, obj_name, thresh=0.05):
        """If gripper site is near the object, kinematically attach it."""
        site = self.site_pos(side)
        op = self.obj_pos(obj_name)
        if np.linalg.norm(site - op) < thresh:
            self.carry_obj = obj_name
            self.carry_side = side
            self.carry_off = op - site
            return True
        return False

    def release_obj(self):
        self.carry_obj = None

    def _carry_update(self):
        sid = self.site[self.carry_side]
        adr = self.obj_jadr[self.carry_obj]
        self.data.qpos[adr:adr + 3] = self.data.site_xpos[sid] + self.carry_off
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        vadr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                              f"{self.carry_obj}_joint")]
        self.data.qvel[vadr:vadr + 6] = 0

    def render_main(self):
        self.renderer.update_scene(self.data)
        return self.renderer.render().copy()

    # ---------- geometry ----------
    def site_pos(self, side, data=None):
        d = data or self.data
        return d.site_xpos[self.site[side]].copy()

    def obj_pos(self, name, data=None):
        d = data or self.data
        return d.xpos[self.obj_body[name]].copy()

    def bin_pos(self, name, data=None):
        d = data or self.data
        return d.xpos[self.bin_body[name]].copy()

    def in_bin(self, obj_name, bin_name):
        p = self.obj_pos(obj_name)
        b = self.bin_pos(bin_name)
        return (abs(p[0] - b[0]) < 0.055 and abs(p[1] - b[1]) < 0.055
                and p[2] < 0.10)

    # ---------- IK on scratch data ----------
    def ik(self, side, target_pos, seed_q=None, iters=1500, alpha=0.6,
           lam=0.08, tol=2e-3):
        """Damped least-squares position IK for `side` arm. Returns 5 joint
        values (arm only, gripper untouched) or None on failure."""
        plan = self._plan
        plan.qpos[:] = self.data.qpos
        plan.qvel[:] = 0
        if seed_q is not None:
            plan.qpos[self.qadr[side][:5]] = seed_q
        qadr5 = self.qadr[side][:5]
        dadr5 = self.dadr[side][:5]
        for _ in range(iters):
            mujoco.mj_forward(self.model, plan)
            err = np.asarray(target_pos) - plan.site_xpos[self.site[side]]
            if np.linalg.norm(err) < tol:
                return plan.qpos[qadr5].copy()
            J = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, plan, J, None, self.site[side])
            Ja = J[:, dadr5]
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + lam ** 2 * np.eye(3), err)
            q = plan.qpos[qadr5] + alpha * dq
            plan.qpos[qadr5] = np.clip(q, self.joint_lo[qadr5],
                                       self.joint_hi[qadr5])
        return None
