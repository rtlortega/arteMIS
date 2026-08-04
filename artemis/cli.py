"""Command-line entry point for the arteMIS optimisation pipeline.

    artemis-run --config config.yaml
    artemis-run --config config.yaml --mode target --scores modcosine

`--mode` / `--scores` override config.yaml for a single run without editing
the file; everything else (paths, sampling space, optimisation metrics)
comes from config.yaml. Default mode is 'global' (no target class needed).
"""
import argparse

from artemis.pipeline.config import VALID_MODES, load_config
from artemis.pipeline.runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="artemis-run",
        description="Run the arteMIS network-optimisation pipeline.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=None,
        help="Override `mode` from the config. Defaults to the config value, or 'global' if unset.",
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="Comma-separated subset of the `scores` keys defined in the config, e.g. 'modcosine,cosine'.",
    )
    parser.add_argument(
        "--n-networks",
        type=int,
        default=None,
        help="Override sampling.n_networks from the config.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(
        args.config,
        mode_override=args.mode,
        scores_override=args.scores,
        n_networks_override=args.n_networks,
    )
    run(cfg, args.config)


if __name__ == "__main__":
    main()
