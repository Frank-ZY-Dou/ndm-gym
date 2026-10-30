"""Learned differentiable dynamics s_{t+1} = f(s_t, a_t, xi) over the state layout of :mod:`ndm_gym.state`.

Every world model exposes ``inputs(S, a, xi_n)``, ``targets(S, S1)`` (regression targets, used to fit the output
normalizer), ``step(S, a, xi_n)``, and the running normalizers ``in_norm`` / ``out_norm``.
"""
