"""Command-line entry point for the arteMIS optimisation pipeline.

    artemis-run --mgf spectra.mgf --chem classes.csv --out results/
    artemis-run --mode seed --mgf spectra.mgf --chem annotations.csv --out results/
    artemis-run --target-class melleolides --mgf spectra.mgf --chem classes.csv --out results/

Inputs, output, mode, scores and chem-level describe a single *run* and are
passed on the command line. How the pipeline behaves (sampling space,
optimisation metrics, score tolerances) is the *method*, and lives in a mode
config under artemis/configs/. Each mode loads its own config by default;
pass --config to use a customised method file instead.
"""
import argparse
import sys
from pathlib import Path

from artemis.pipeline.config import (
    DEFAULT_TOLERANCE,
    KNOWN_SCORES,
    VALID_MODES,
    load_config,
)
from artemis.pipeline.runner import run

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CHEM_LEVEL = "npc_pathway_results"


class ArtemisArgumentParser(argparse.ArgumentParser):
    """argparse parser with a friendlier hint for unquoted multi-word values."""

    def error(self, message: str) -> None:
        if "unrecognized arguments" in message:
            message += (
                "\n\nIf a value contains spaces (e.g. a class name), wrap it in "
                "quotes, otherwise the shell splits it into several arguments:\n"
                '  --target-class "Shikimates and Phenylpropanoids"'
            )
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = ArtemisArgumentParser(
        description="Run the arteMIS network-optimisation pipeline.",
    )

    # --- Inputs / output (this run) ---
    parser.add_argument("--mgf", required=True, help="Input spectra (.mgf).")
    parser.add_argument(
        "--chem",
        required=True,
        help="Chemical info table (csv/tsv): class labels, and SMILES unless --smiles is given.",
    )
    parser.add_argument(
        "--smiles",
        default=None,
        help="Optional separate SMILES table (csv/tsv), merged on feature id. "
        "Use for SIRIUS-style inputs; omit when SMILES are already in --chem (MS2Query).",
    )
    parser.add_argument("--out", required=True, help="Output directory.")

    # --- Run behaviour ---
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="global",
        help="Analysis mode (default: %(default)s). Each mode loads its own method config.",
    )
    parser.add_argument(
        "--target-class",
        default=None,
        help="Class to target. Implies --mode target.",
    )
    parser.add_argument(
        "--chem-level",
        default=DEFAULT_CHEM_LEVEL,
        help="Column in --chem used for the chemical optimisation "
        "(default: %(default)s). MS2Query → 'npc_pathway_results'; SIRIUS/CANOPUS → 'NPC#pathway'.",
    )
    parser.add_argument(
        "--scores",
        default="modcosine",
        help="Comma-separated score families to compute (default: %(default)s). "
        f"Available: {', '.join(KNOWN_SCORES)}.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="m/z matching tolerance for the score families (default: %(default)s).",
    )
    parser.add_argument(
        "--n-networks",
        type=int,
        default=100,
        help="Number of candidate networks to sample and evaluate (default: %(default)s).",
    )

    # --- Method config (optional override) ---
    parser.add_argument(
        "--config",
        default=None,
        help="Custom method config (yaml). Defaults to artemis/configs/<mode>.yaml.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # --target-class implies target mode; guard against a contradictory --mode.
    mode = args.mode
    if args.target_class:
        if mode not in ("global", "target"):
            build_parser().error(
                f"--target-class is only valid with --mode target, not --mode {mode}"
            )
        mode = "target"

    config_path = Path(args.config) if args.config else CONFIGS_DIR / f"{mode}.yaml"
    if not config_path.is_file():
        build_parser().error(f"method config not found: {config_path}")

    print(f"Optimizing chemistry on column: {args.chem_level!r}")
    print(
        "  (MS2Query inputs -> 'npc_pathway_results';  SIRIUS/CANOPUS -> 'NPC#pathway')"
    )

    cfg = load_config(
        str(config_path),
        mode=mode,
        mgf_file=args.mgf,
        chem_file=args.chem,
        smiles_file=args.smiles,
        out_dir=args.out,
        chem_level=args.chem_level,
        target_class=args.target_class,
        scores=args.scores,
        tolerance=args.tolerance,
        n_networks=args.n_networks,
    )
    run(cfg, str(config_path))


if __name__ == "__main__":
    sys.exit(main())
