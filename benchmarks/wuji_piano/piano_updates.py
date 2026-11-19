"""NDM Gym hand updates for benchmarks whose measured piano state is one flat tensor."""

from types import SimpleNamespace

import torch

from ndm_gym.algos.hand_algos import HandSuperMimic, HandSuperTrack
from ndm_gym.algos.hand_fast import optimizer
from ndm_gym.algos.supertrack import LossBalancer
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.policies.critic import Critic
from ndm_gym.policies.hand_policy import HandPolicy
from ndm_gym.worldmodels.piano import NUM_ACTUATORS as A, KEY_COUNT, PianoDynamics

D = 2 * A + KEY_COUNT
H = 3
LOOK = 11


def make_replay(capacity, worlds, device, *, state_dim=D, num_actions=A, update_graphs=False):
    """Allocate measured piano transitions in the layout consumed by the hand trainers."""
    fields = {key: (width, torch.float32) for key, width in (("s", state_dim), ("s1", state_dim), ("u", num_actions), ("a_prev", num_actions))}
    fields.update(f=((), torch.long), done=((), torch.bool))
    buffer = ReplayBuffer(capacity, worlds, device, fields=fields, priorities=False)
    buffer.window_keys = ("s", "u", "s1", "done")
    if update_graphs:
        buffer.use_device_ptr(device_size=True)
    return buffer


class PianoReplay:
    rollout = None  # use the task's eager per-step differentiable piano model

    @staticmethod
    def decode_replay_state(state):
        return state


class PianoTrackUpdates(PianoReplay, HandSuperTrack):
    start_fk = None  # st_step computes its contact features from the predicted piano state
    st_objective = HandSuperTrack._st_objective


class PianoMimicUpdates(PianoReplay, HandSuperMimic):
    reuse_input = False
    lap = None
    sm_objective = HandSuperMimic._sm_objective
    critic_targets = HandSuperMimic._critic_targets
    critic_loss = HandSuperMimic._critic_loss


class OdePiano:
    def __init__(self, geometry, roll, *, update_graphs=False):
        self.dev = roll.device
        self.args = SimpleNamespace(
            wm_batches=1, wm_batch=96, wm_window=2, pi_batch=96, lr=0.0008, fused_adam=roll.is_cuda, update_graphs=update_graphs
        )
        self.horizon = H
        self.goal = torch.cat((roll, torch.zeros(LOOK + H, KEY_COUNT, device=self.dev, dtype=torch.bool))).float()
        self.length = len(roll)
        self.future = self.goal.unfold(0, LOOK, 1).transpose(1, 2).reshape(-1, LOOK * KEY_COUNT)
        self.wm = PianoDynamics(geometry).to(self.dev)
        self.pi = HandPolicy(D + A, LOOK * KEY_COUNT, A, hidden=(192, 128)).to(self.dev)
        self.opt_wm = optimizer(self.args, self.dev, self.wm, "wm")
        self.opt_pi = optimizer(self.args, self.dev, self.pi, "pi")
        self.no_sync = update_graphs
        self.task = SimpleNamespace(max_step_reward=1.0)
        self.weights = torch.tensor([1.0, 0.85, 0.20], device=self.dev)

    def action(self, state, prev, f):
        return self.pi(torch.cat((state, prev), -1), self.future[f].expand(len(state), -1))

    def tracking_errors(self, state, target):
        k = state[:, 2 * A :]
        count = target.sum(-1).clamp_min(1)
        due = ((k - 1).abs() * target).sum(-1) / count
        wrong = (k.abs() * (1 - target)).amax(-1)
        dx, dy, dz = self.wm.geom.distances(self.wm.geom(state[:, :A]))
        dist = (dx.square() + dy.square() + dz.square() + 1e-8).amin(1).sqrt()
        reach = (dist * target).sum(-1) / count / 0.035
        return torch.stack((due, wrong, reach), -1)

    def wm_rollout(self, state, actions, truth, mask):
        loss = 0
        for j in range(actions.shape[1]):
            state = self.wm(state, actions[:, j])
            err = (state - truth[:, j]).square()
            depth = truth[:, j, 2 * A :]
            keyerr = err[:, 2 * A :]
            term = (
                err[:, : 2 * A].mean(-1)
                + keyerr.mean(-1)
                + 0.6 * keyerr.amax(-1)
                + (keyerr * depth.clamp(0, 1)).sum(-1) / depth.clamp(0, 1).sum(-1).clamp_min(1)
            )
            loss += (term * mask[:, j]).mean()
        return loss


class OdeSuperTrack(OdePiano, PianoTrackUpdates):
    def __init__(self, geometry, roll, *, update_graphs=False):
        super().__init__(geometry, roll, update_graphs=update_graphs)
        self.imag_noise = self.fail_w = self.reg = 0.0
        self.term_scale = torch.ones(3, device=self.dev)
        self.bal = LossBalancer(3).to(self.dev)
        self.bal.w.copy_(self.weights / H)
        self.bal.ready.fill_(True)

    def st_step(self, state, fk, f, prev, noise):
        act = self.action(state, prev, f)
        state = self.wm(state, act + noise)
        errors = self.tracking_errors(state, self.goal[f])
        return state, None, (f + 1).clamp(max=self.length), act, errors, torch.zeros_like(f, dtype=state.dtype), 1.0 - errors.mean(-1)


class OdeSuperMimic(OdePiano, PianoMimicUpdates):
    def __init__(self, geometry, roll, *, update_graphs=False):
        super().__init__(geometry, roll, update_graphs=update_graphs)
        self.args.gamma, self.args.lam = 0.99, 0.95
        self.args.critic_epochs, self.args.critic_minibatches = 5, 4
        self.critic = Critic(D + A + LOOK * KEY_COUNT).to(self.dev)
        self.opt_v = optimizer(self.args, self.dev, self.critic, "critic")

    def start_features(self, state, f, prev):
        return None, torch.cat((state, prev, self.future[f]), -1)

    def imag_step(self, state, fk, f, prev):
        act = self.action(state, prev, f)
        state = self.wm(state, act)
        reward = 1.0 - (self.tracking_errors(state, self.goal[f]) * self.weights).sum(-1)
        done = (f >= self.length - 1).to(state.dtype)
        f = (f + 1).clamp(max=self.length)
        _, x = self.start_features(state, f, act)
        return state, None, f, act, reward, done, x
