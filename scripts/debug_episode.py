"""Debug: run one episode, snapshot after every waypoint + obj z trace."""
import sys, os
import numpy as np
import imageio.v3 as iio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.bimanual_env import BimanualSortEnv, GRIP_OPEN, GRIP_CLOSED
from scripts.gen_demos import execute_waypoint, pick_arm, TASKS

env = BimanualSortEnv()
env.reset(123)
task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
obj_name, bin_name, instr = TASKS[task_id][:3]
op = env.obj_pos(obj_name)
side = pick_arm(env, op)
print(f"task={instr} obj@{np.round(op,3)} arm={side}")

shots = []
steps = [
    ("hover",  op + [0, 0, 0.10],  GRIP_OPEN,   1.0),
    ("descend",op + [0, 0, 0.012], GRIP_OPEN,   0.8),
    ("close",  op + [0, 0, 0.012], GRIP_CLOSED, 0.7),
    ("lift",   op + [0, 0, 0.14],  GRIP_CLOSED, 1.0),
    ("carry",  env.bin_pos(bin_name) + [0, 0, 0.14], GRIP_CLOSED, 1.2),
    ("release",env.bin_pos(bin_name) + [0, 0, 0.10], GRIP_OPEN,   0.7),
]
for name, tgt, grip, secs in steps:
    ok = execute_waypoint(env, side, tgt, grip, secs, False, [])
    p = env.obj_pos(obj_name)
    print(f"{name}: ik_ok={ok} obj_pos={np.round(p,3)} grip_q={env.data.qpos[env.qadr[side][5]]:.2f}")
    shots.append((name, env.render_main(), env.obs()["images"][f"{side}_wrist_cam"]))

# tile the shots
row = np.concatenate([np.concatenate([s[1], s[2]], axis=0) for s in shots[:3]], axis=1)
row2 = np.concatenate([np.concatenate([s[1], s[2]], axis=0) for s in shots[3:]], axis=1)
iio.imwrite("results/debug_waypoints.png", np.concatenate([row, row2], axis=1))
print("saved results/debug_waypoints.png")
print("final in_bin:", env.in_bin(obj_name, bin_name))
