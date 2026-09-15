"""VoiceSort front end: voice/text command -> VLA rollout on Intel OpenVINO.

Modes:
  --text "put the red cube in the left bin"     typed instruction
  --wav  command.wav                            Speechmatics batch STT
                                                (needs SPEECHMATICS_API_KEY)
  --device CPU|GPU.0                            OpenVINO device (default CPU)

The transcript is parsed to (object, bin) by keyword rules; the raw
instruction text is embedded with MiniLM and fed to the policy, so
rephrased commands work too. Unrecognized commands are declined honestly.
"""
import os, sys, json, argparse
import numpy as np
import imageio.v3 as iio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.bimanual_env import BimanualSortEnv
from scripts.gen_demos import TASKS
from scripts.deploy_openvino import OVPolicy, rollout
from policy.text_embed import embed_texts
import torch

OBJ_WORDS = {"red": "obj_red_cube", "cube": "obj_red_cube",
             "blue": "obj_blue_sphere", "sphere": "obj_blue_sphere",
             "green": "obj_green_cyl", "cylinder": "obj_green_cyl"}
BIN_WORDS = {"left": "bin_left", "right": "bin_right"}


def transcribe(wav_path):
    """Speechmatics batch transcription. Requires SPEECHMATICS_API_KEY."""
    key = os.environ.get("SPEECHMATICS_API_KEY")
    if not key:
        raise RuntimeError("SPEECHMATICS_API_KEY not set")
    from speechmatics.batch_client import BatchClient
    from speechmatics.models import (BatchTranscriptionConfig,
                                     ConnectionSettings)
    settings = ConnectionSettings(
        url="https://asr.api.speechmatics.com/v2", auth_token=key)
    with BatchClient(settings) as client:
        job_id = client.submit_job(
            wav_path,
            transcription_config=BatchTranscriptionConfig(language="en"))
        return client.wait_for_completion(job_id, transcription_format="txt")


def parse_command(text):
    """Keyword rules -> (obj_name, bin_name). Returns None to decline."""
    t = text.lower()
    obj = next((OBJ_WORDS[w] for w in ["red", "cube", "blue", "sphere",
                                       "green", "cylinder"] if w in t), None)
    b = next((BIN_WORDS[w] for w in ["left", "right"] if w in t), None)
    # weak disambiguation: 'cube' alone implies red, 'sphere' implies blue
    if obj is None or b is None:
        return None
    return obj, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--wav", default=None)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--ir", default="ckpt/openvino/model_fp32.xml")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    if args.wav:
        text = transcribe(args.wav)
        print(f"transcript: '{text}'")
    elif args.text:
        text = args.text
    else:
        text = input("command> ")
        print(f"typed: '{text}'")

    parsed = parse_command(text)
    if parsed is None:
        print(f"declining: could not map '{text}' to a known object/bin")
        return
    obj_name, bin_name = parsed

    # matching canonical task gives the designated arm + spawn side
    match = [t for t in TASKS.values()
             if t[0] == obj_name and t[1] == bin_name]
    side = match[0][3] if match else ("left" if bin_name == "bin_left"
                                     else "right")
    print(f"parsed -> {obj_name} -> {bin_name} (arm={side})")

    emb = torch.from_numpy(embed_texts([text]))[0]
    env = BimanualSortEnv()
    env.grasp_assist = True
    env.assist_obj = obj_name
    env.reset(args.seed, spawn_side=side, spawn_obj=obj_name)
    pol = OVPolicy(args.ir, args.device)
    frames = rollout(env, pol, emb, max_steps=250, video=args.video)
    ok = env.in_bin(obj_name, bin_name)
    print(f"result: {'SUCCESS' if ok else 'FAIL'} "
          f"({obj_name} -> {bin_name} via OpenVINO/{args.device})")
    if args.video and frames:
        p = "results/voice_demo.mp4"
        iio.imwrite(p, np.stack(frames), fps=20)
        print(f"video -> {p}")


if __name__ == "__main__":
    main()
