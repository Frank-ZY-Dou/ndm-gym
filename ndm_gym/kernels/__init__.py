"""Hand-written GPU kernels of the world models (Triton). Opt-in: the library imports them only where a flag asks for them.

rf_block: RigidFormer's Transformer block (``ndm_gym.worldmodels.rigidformer.Block``) as one fused forward kernel, and
``fused_copy`` that puts it into a network or world model for inference.
rf_train: the block's forward and backward passes for training (``RigidFormerNet(fused_train=True)``, ``--rf_fused_train``).
"""
