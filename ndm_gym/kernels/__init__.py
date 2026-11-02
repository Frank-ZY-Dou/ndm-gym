"""Hand-written GPU kernels of the world models (Triton). Opt-in: the library imports them only where a flag asks for them.

rf_train: RigidFormer's Transformer block (``ndm_gym.worldmodels.rigidformer.Block``) forward and backward passes for
training (``RigidFormerNet(fused_train=True)``, ``--rf_fused_train``).
"""
