"""Command-line defaults shared by piano trainers with checkpoint evaluation modes."""


def training_defaults(parser, args, *, training, **defaults):
    """Apply training defaults after parsing so explicitly supplied fitting options cannot be ignored."""
    supplied = defaults.keys() & vars(args).keys()
    if not training and supplied:
        options = ", ".join("--" + key.replace("_", "-") for key in sorted(supplied))
        parser.error(f"Training options cannot be used during evaluation: {options}")
    for key, value in defaults.items():
        vars(args).setdefault(key, value)
