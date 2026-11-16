"""Full Hand control-step schedule contract for a future native Door runner.

Independent of the legacy episode-round runner. Counts alone do not establish
algorithm equivalence: collection, normalization and learner semantics must also
match the declared Door adaptation before formal execution is admitted.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FullHandSchedule:
    iterations: int = 20_000
    worlds: int = 2048
    warmup_steps: int = 32
    wm_batches: int = 4
    wm_only_iters: int = 300
    critic_epochs: int = 5
    critic_minibatches: int = 4

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if min(self.iterations, self.worlds, self.wm_batches,
               self.critic_epochs, self.critic_minibatches) == 0:
            raise ValueError('training dimensions must be positive')
        if self.wm_only_iters > self.iterations:
            raise ValueError('WM-only phase exceeds training budget')

    def policy_due(self, iteration):
        if type(iteration) is not int or not 1 <= iteration <= self.iterations:
            raise ValueError('iteration must be within the one-based training schedule')
        return iteration > self.wm_only_iters

    def counts(self, completed=0, *, method):
        if method not in ('supertrack', 'supermimic'):
            raise ValueError('unsupported method')
        if type(completed) is not int or not 0 <= completed <= self.iterations:
            raise ValueError('completed must be a valid iteration boundary')
        policy = max(0, completed - self.wm_only_iters)
        return dict(training_transitions=completed * self.worlds,
                    warmup_transitions=self.warmup_steps * self.worlds,
                    wm_optimizer_steps=completed * self.wm_batches,
                    policy_updates=policy,
                    critic_optimizer_steps=(policy * self.critic_epochs * self.critic_minibatches
                                            if method == 'supermimic' else 0))

    def assert_counts(self, actual, completed, *, method):
        expected = self.counts(completed, method=method)
        if actual != expected:
            raise ValueError(f'schedule mismatch: expected {expected}, got {actual}')
