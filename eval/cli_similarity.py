import argparse
import logging

import torch as th

from .config import Config, MODEL_LIST
from .data import get_datasets_and_splits
from .encoder import TransformerEncoder
from .similarity import create_sense_vectors, evaluate_similarity


def _resolve_modals(args):
    if args.modal == "both":
        return ["may", "must"]
    return [args.modal]


def _validate_paths(args, modals):
    if "may" in modals and not args.may_data_path:
        raise SystemExit("--may-data-path is required when --modal includes 'may'")
    if "must" in modals and not args.must_data_path:
        raise SystemExit("--must-data-path is required when --modal includes 'must'")


def _modal_data_path(args, modal):
    return args.may_data_path if modal == "may" else args.must_data_path


def main():
    parser = argparse.ArgumentParser(
        description="Modal sense disambiguation: similarity-based evaluation"
    )
    parser.add_argument("--may-data-path", default=None)
    parser.add_argument("--must-data-path", default=None)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--models", nargs="+", default=MODEL_LIST)
    parser.add_argument("--modal", choices=["may", "must", "both"], default="both")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--layers", nargs="+", type=int, default=[-1, -2, -3, -4])
    parser.add_argument("--layer-op", default="sum")
    parser.add_argument("--subword-op", default="mean")
    parser.add_argument("--mode", default="regular")
    parser.add_argument("--seed", type=int, default=199)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    modals = _resolve_modals(args)
    _validate_paths(args, modals)

    config = Config(
        results_path=args.results_path,
        max_seq_len=args.max_seq_len,
        layers=args.layers,
        layer_op=args.layer_op,
        subword_op=args.subword_op,
        mode=args.mode,
        seed=args.seed,
    )

    per_modal_splits = {
        modal: get_datasets_and_splits(_modal_data_path(args, modal))
        for modal in modals
    }

    for model_name in args.models:
        try:
            encoder = TransformerEncoder(model_name, config)
        except Exception as e:
            logging.error(f"Failed to load {model_name}: {e}")
            continue

        for modal in modals:
            base_path = _modal_data_path(args, modal)
            for dataset, splits in per_modal_splits[modal].items():
                for split in splits:
                    try:
                        sv_path = create_sense_vectors(
                            encoder,
                            model_name,
                            modal,
                            base_path,
                            dataset,
                            split,
                            args.results_path,
                            config,
                        )
                        accs, _ = evaluate_similarity(
                            encoder,
                            sv_path,
                            model_name,
                            modal,
                            base_path,
                            dataset,
                            split,
                            args.results_path,
                            config,
                        )
                        logging.info(
                            f"{model_name} {modal}/{dataset}/{split}: {accs}"
                        )
                    except Exception as e:
                        logging.error(
                            f"Error {model_name} {modal}/{dataset}/{split}: {e}"
                        )

        del encoder
        if th.cuda.is_available():
            th.cuda.empty_cache()


if __name__ == "__main__":
    main()