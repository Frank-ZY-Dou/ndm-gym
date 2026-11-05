# Muscle-actuated character

MuscleVAE's character: 23 bodies, 284 Hill-type muscles with fatigue, on Newton with MuJoCo-Warp directly (no Isaac
Lab; `ndm_gym/sim/rigid/muscle_sim.py`), tracking MuscleVAE's three LAFAN1 clips. The world models predict the
character's state and the muscles' fatigue; the RigidFormer has a token per body and per muscle
(`--rf_muscle_tokens all`) or per body, with the muscles' wrench and pooled encodings (`--rf_muscle_tokens body`).

```bash
bash tools/download_musclevae.sh && python tools/convert_musclevae.py   # character, muscles, motions (not redistributed)
python -m ndm_gym.train_muscles --algo supertrack --log_dir logs/mus_st
python -m ndm_gym.train_muscles --algo supermimic --log_dir logs/mus_sm

# MuscleVAE's released policy in this simulator, rendered in their look (bash tools/download_mass_meshes.sh for the bones)
bash tools/download_musclevae_pretrained.sh
python tools/replay_musclevae.py --out_dir replay
MUJOCO_GL=egl python tools/render_musclevae.py walk.mp4 --states replay/states_lafan_walk_0.npz --ref
```

Selected configurations and results: [README](../../README.md#muscle-actuated-character), [`results.md`](../../benchmarks/musclevae/results.md). The character and the
models: [design](../musclevae_design.md).
