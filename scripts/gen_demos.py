"""Generate scripted demonstrations for bimanual sorting.

Per episode: pick (object, bin) task from the language task table, choose the
arm nearest the object, execute a waypoint sequence (approach/grasp/lift/
carry/release/retreat). IK plans each waypoint on scratch data; the position
actuators execute it under real physics so grasps/contacts are genuine.
Frames (3 cams + 14-dim state + 14-dim action) are written to a LeRobot
dataset with the natural-language task string per episode.

IK is used ONLY to script demonstrations -- the learned policy drives control
at inference; no IK runs at deploy time.
"""
import os, sys, time
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.bimanual_env import BimanualSortEnv, HOME, GRIP_OPEN, GRIP_CLOSED

# task_id -> (object body, bin body, instruction string, designated arm)
# Each task has ONE serving arm; the target object always spawns on that
# arm's side, so the instruction (not arm-choice ambiguity) drives behavior.
TASKS = {
    0: ("obj_red_cube", "bin_left", "put the red cube in the left bin", "left"),
    1: ("obj_blue_sphere", "bin_right", "put the blue sphere in the right bin", "right"),
    2: ("obj_green_cyl", "bin_left", "put the green cylinder in the left bin", "left"),
}
ARM_MOUNT_X = {"left": -0.16, "right": 0.16}


def pick_arm(env, obj_pos):
    return "left" if obj_pos[0] <= 0 else "right"


def execute_waypoint(env, side, target, gripper, seconds, record, ep_frames,
                     ramp_from=None):
    """IK to target on scratch data, then drive actuator to it over `seconds`.
    Returns success bool."""
    q_arm = env.ik(side, target, seed_q=None)
    if q_arm is None:
        return False
    cur = env.data.ctrl.copy()
    tgt = cur.copy()
    tgt[env.act_id[side][:5]] = q_arm
    tgt[env.act_id[side][5]] = gripper
    n = max(1, int(round(seconds / env.control_dt)))
    for i in range(n):
        a = cur + (tgt - cur) * (i + 1) / n
        obs = env.step(a)
        if record:
            ep_frames.append({
                "observation.images.overhead": obs["images"]["overhead_cam"],
                "observation.images.left_wrist": obs["images"]["left_wrist_cam"],
                "observation.images.right_wrist": obs["images"]["right_wrist_cam"],
                "observation.state": obs["state"],
                "action": a.astype(np.float32).copy(),
            })
    return True


def run_episode(env, task_id, record=False):
    """Execute one scripted episode. Returns (frames, success)."""
    obj_name, bin_name, instruction, side = TASKS[task_id]
    ep_frames = []
    rec = lambda *a, **k: None

    op = env.obj_pos(obj_name)
    other = "right" if side == "left" else "left"
    g = {"open": GRIP_OPEN, "closed": GRIP_CLOSED}

    def wp(target, grip, secs):
        return execute_waypoint(env, side, target, grip, secs, record, ep_frames)

    # sequence
    if not wp(op + [0, 0, 0.10], g["open"], 1.0):  return ep_frames, False
    if not wp(op + [0, 0, 0.012], g["open"], 0.8): return ep_frames, False
    if not wp(op + [0, 0, 0.012], g["closed"], 0.7): return ep_frames, False
    if not env.grasp_obj(side, obj_name):
        return ep_frames, False
    if not wp(op + [0, 0, 0.14], g["closed"], 1.0): return ep_frames, False
    bp = env.bin_pos(bin_name)
    if not wp(bp + [0, 0, 0.14], g["closed"], 1.2): return ep_frames, False
    env.release_obj()
    if not wp(bp + [0, 0, 0.10], g["open"], 0.7): return ep_frames, False
    if not wp(bp + [0, 0, 0.18], g["open"], 0.8): return ep_frames, False
    # park the used arm back at HOME, leave the other parked
    park = np.concatenate([HOME[:6], HOME[6:]]) if False else None
    # return to HOME via interpolation
    cur = env.data.ctrl.copy()
    tgt = cur.copy()
    tgt[:] = HOME
    n = int(round(1.0 / env.control_dt))
    for i in range(n):
        a = cur + (tgt - cur) * (i + 1) / n
        obs = env.step(a)
        if record:
            ep_frames.append({
                "observation.images.overhead": obs["images"]["overhead_cam"],
                "observation.images.left_wrist": obs["images"]["left_wrist_cam"],
                "observation.images.right_wrist": obs["images"]["right_wrist_cam"],
                "observation.state": obs["state"],
                "action": a.astype(np.float32).copy(),
            })
    # settle, then success check
    for _ in range(20):
        mujoco.mj_step(env.model, env.data)
    return ep_frames, env.in_bin(obj_name, bin_name)


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out_dir = "demo_data"
    env = BimanualSortEnv()

    # dry-run one episode unrecorded to validate
    env.reset(123, spawn_side=TASKS[0][3], spawn_obj=TASKS[0][0])
    frames, ok = run_episode(env, 0, record=True)
    print(f"validation episode: success={ok}, frames={len(frames)}")
    import imageio.v3 as iio
    iio.imwrite("results/demo_last.png", env.render_main())

    if not ok:
        print("validation FAILED - tune grasp before mass generation")
        return

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    if os.path.exists(out_dir):
        import shutil; shutil.rmtree(out_dir)
    ds = LeRobotDataset.create(
        repo_id="voicesort_so101", root=out_dir, robot_type="so101_bimanual",
        fps=20,
        features={
            "observation.images.overhead": {"dtype": "image", "shape": (256, 256, 3), "names": ["h", "w", "c"]},
            "observation.images.left_wrist": {"dtype": "image", "shape": (256, 256, 3), "names": ["h", "w", "c"]},
            "observation.images.right_wrist": {"dtype": "image", "shape": (256, 256, 3), "names": ["h", "w", "c"]},
            "observation.state": {"dtype": "float32", "shape": (12,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (12,), "names": ["action"]},
        },
        image_writer_threads=8, image_writer_processes=4,
    )
    n_ok = 0
    t0 = time.time()
    for ep in range(n_eps):
        task_id = ep % len(TASKS)
        env.reset(1000 + ep, spawn_side=TASKS[task_id][3],
                  spawn_obj=TASKS[task_id][0])
        frames, ok = run_episode(env, task_id, record=True)
        if ok:
            for f in frames:
                ds.add_frame(f, task=TASKS[task_id][2])
            ds.save_episode()
            n_ok += 1
        print(f"ep {ep}: task={task_id} ok={ok} frames={len(frames)} "
              f"({time.time()-t0:.0f}s)")
    ds.encode_videos()
    print(f"done: {n_ok}/{n_eps} successful episodes -> {out_dir}")


if __name__ == "__main__":
    main()
