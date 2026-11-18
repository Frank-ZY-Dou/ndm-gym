"""Render saved GPU poses with audio generated from detected keys."""

import argparse
import itertools
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train_fur_elise import DT, KEY_DOWN_FRACTION, LOWEST_PITCH, MUSIC_FRAMES, WARMUP, build


def audio_from_keys(pressed, path):
    rate = 44100
    samples = round(DT * rate)
    sound = np.zeros(len(pressed) * samples)
    for key, track in enumerate(pressed.T):
        pos = 0
        for down, group in itertools.groupby(track):
            length = sum(1 for _ in group) * samples
            if down:
                t = np.arange(min(length + round(0.12 * rate), len(sound) - pos)) / rate
                pitch = 440 * 2 ** ((key + LOWEST_PITCH - 69) / 12)
                tone = sum(gain * np.sin(2 * np.pi * pitch * harmonic * t) for harmonic, gain in ((1, 1), (2, 0.36), (3, 0.18), (4, 0.08)))
                env = (1 - np.exp(-t / 0.004)) * np.exp(-t / 1.2) * np.exp(-np.maximum(0, t - length / rate) / 0.032)
                sound[pos : pos + len(t)] += tone * env
            pos += length
    with wave.open(str(path), "wb") as file:
        file.setparams((1, 2, rate, 0, "NONE", "none"))
        file.writeframes((sound / max(1.0, np.abs(sound).max()) * 28000).astype("<i2").tobytes())


def note_name(pitch):
    return ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")[pitch % 12] + str(pitch // 12 - 1)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from render import draw_mark

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--hand-dir", required=True, type=Path)
    parser.add_argument("--keyboard", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--song", choices=("fur_elise", "five_finger_scale", "ode_to_joy"), default="fur_elise")
    args = parser.parse_args()
    trace = np.load(args.rollout)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    if args.song in ("five_finger_scale", "ode_to_joy"):
        if args.song == "ode_to_joy":
            from ode_to_joy import build as build_song

            title = "Ode to Joy  |  Eden · 90 BPM"
            cam.lookat[:] = [0.06, -0.13, 0.050]
            cam.distance = 0.72
        else:
            from five_finger_scale import build as build_song

            title = "Five-finger scale  |  Eden · 80 BPM"
            cam.lookat[:] = [0.06, -0.012, 0.048]
            cam.distance = 0.54
        model = build_song(args.hand_dir, args.keyboard)
        warmup, end = 20, len(trace["goal"])
        due_keys = [set(np.flatnonzero(mask) + LOWEST_PITCH) for mask in trace["goal"]]
        cam.azimuth, cam.elevation = 179, -48
        opt.sitegroup[:] = 0
    else:
        model = build(args.hand_dir, args.keyboard)
        title = "Für Elise  |  pickup and bars 1–5"
        warmup, end = WARMUP, WARMUP + MUSIC_FRAMES
        due_keys = [{int(k) for k in parts if k} for parts in trace["parts"]]
        cam.lookat[:] = [0.02, -0.075, 0.077]
        cam.distance, cam.azimuth, cam.elevation = 0.86, 179, -36
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    model.vis.headlight.ambient[:] = [0.55] * 3
    data = mujoco.MjData(model)
    opt.geomgroup[2] = 0
    pressed = trace["depth"] >= KEY_DOWN_FRACTION
    geoms = [model.geom(f"key_{k}_visual") for k in range(88)]
    colors = [g.rgba.copy() for g in geoms]
    encoder = shutil.which("ffmpeg")
    if not encoder:
        import imageio_ffmpeg

        encoder = imageio_ffmpeg.get_ffmpeg_exe()
    fonts = {size: ImageFont.truetype("DejaVuSans.ttf", size) for size in (17, 21, 27)}
    with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, height=540, width=960) as renderer:
        audio = Path(tmp) / "keys.wav"
        audio_from_keys(pressed, audio)
        command = [
            encoder,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "960x540",
            "-r",
            "20",
            "-i",
            "-",
            "-i",
            str(audio),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(args.out),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
            for frame, (pose, down, due) in enumerate(zip(trace["qpos"], pressed, due_keys, strict=True)):
                data.qpos[:] = pose
                mujoco.mj_kinematics(model, data)
                played = set(np.flatnonzero(down) + LOWEST_PITCH)
                for k, g in enumerate(geoms):
                    pitch = k + LOWEST_PITCH
                    if pitch in played:
                        g.rgba[:] = (0.20, 0.82, 0.51, 1) if pitch in due else (0.95, 0.37, 0.24, 1)
                    elif pitch in due:
                        g.rgba[:] = (0.27, 0.57, 0.94, 1)
                    else:
                        g.rgba[:] = colors[k]
                renderer.update_scene(data, camera=cam, scene_option=opt)
                image = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(image)
                draw.text((20, 16), title, font=fonts[27], fill="white")
                score_time = (frame - warmup) * DT
                label = "Lead-in" if frame < warmup else "Release" if frame >= end else f"{score_time:.2f} s"
                draw.text((20, 54), f"Wuji Hand 2   ·   MuJoCo-Warp   ·   {label}", font=fonts[17], fill="#bdc8d1")
                expected = " + ".join(note_name(int(v)) for v in sorted(due)) or "Rest"
                actual = " + ".join(note_name(int(v)) for v in sorted(played)) or "Rest"
                draw.rectangle((0, 486, 960, 540), fill=(21, 28, 35))
                draw.text((20, 500), f"Score   {expected}", font=fonts[21], fill="#9dc6ff")
                draw.text((390, 500), f"Played   {actual}", font=fonts[21], fill="#aee8c7")
                draw_mark(draw, 960, 540, fonts[17], fill="#c1c9d0")
                proc.stdin.write(image.tobytes())
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("Video encoding failed")


if __name__ == "__main__":
    main()
