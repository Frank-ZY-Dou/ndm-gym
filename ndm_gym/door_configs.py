"""Door capacity adaptation of measured Hand accurate settings.

This is a native Adroit adaptation, not an upstream configuration or a measured
Door baseline. The collector, prior, goal, normalization and terminal semantics
still differ. See benchmarks/adroit_door/config_audit before running.
"""
import sys

NAME = 'door-hand-accurate-capacity-v1'
UNAVAILABLE = ('reset-masked', 'no-sync', 'compile-eager', 'collect-graph', 'update-graphs',
               'imag-hoist', 'pi-deferred-grads', 'wm-deferred-grads', 'muon-batched',
               'wm-presample', 'layout')


def parse_args(parser, argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    probe = type(parser)(add_help=False)
    probe.add_argument('--config')
    probe.add_argument('--method')
    requested, _ = probe.parse_known_args(argv)
    parser.add_argument('--config', choices=(NAME,))
    for option in UNAVAILABLE:
        parser.add_argument('--' + option, nargs='?', const=True, default=None,
                            help='Unavailable in native Door; an explicit request is an error')
    if requested.config == NAME:
        parser.set_defaults(horizon=16, wm_hidden=[1024, 1024, 1024],
                            policy_hidden=[1024, 512, 256], wm_batch_size=2048,
                            batch_size=4096, wm_window=8, rounds=20000,
                            bootstrap_updates=1200, wm_updates=4, policy_updates=1,
                            policy_lr=4e-4 if requested.method == 'supertrack' else 2e-4,
                            learning_rate=4e-4 if requested.method == 'supertrack' else 2e-4,
                            critic_epochs=5, critic_minibatches=4,
                            upstream_policy_optimizer=True,
                            task_weights=requested.method == 'supertrack')
    args = parser.parse_args(argv)
    for option in UNAVAILABLE:
        if getattr(args, option.replace('-', '_')) is not None:
            parser.error(f'--{option} is unavailable in native Door; Hand/XMimic support does not imply Door support')
    return args
