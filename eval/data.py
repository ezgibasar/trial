import os
import logging

from .config import MODAL_MAP, COARSE_SENSES


def get_datasets_and_splits(base_path):
    if not os.path.exists(base_path):
        logging.error(f"Path not found: {base_path}")
        return {}

    datasets_splits = {}
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.isdigit():
            splits = sorted(
                [
                    d
                    for d in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()
                ],
                key=int,
            )
            datasets_splits[os.path.basename(os.path.normpath(base_path))] = splits
            break
        splits = [
            d
            for d in os.listdir(entry_path)
            if os.path.isdir(os.path.join(entry_path, d)) and d.isdigit()
        ]
        if splits:
            datasets_splits[entry] = sorted(splits, key=int)

    return datasets_splits


def load_coarse_senses(modal):
    return {modal: list(COARSE_SENSES[modal])}


def get_word_classes(word, modal):
    return dict(MODAL_MAP[modal])


def _resolve_data_paths(base_path, dataset, split_num, modal, split):
    data_file = f"{modal}_{split}.data.txt"
    gold_file = f"{modal}_{split}.gold.txt"

    candidates = []
    if dataset is not None and split_num is not None:
        candidates.append(os.path.join(base_path, dataset, str(split_num)))
    if split_num is not None:
        candidates.append(os.path.join(base_path, str(split_num)))
    if dataset is not None:
        candidates.append(os.path.join(base_path, dataset))
    candidates.append(base_path)

    for candidate in candidates:
        data_path = os.path.join(candidate, data_file)
        if os.path.exists(data_path):
            return data_path, os.path.join(candidate, gold_file)
    return None, None


def load_instances(
    word,
    split,
    modal,
    base_path,
    dataset=None,
    split_num=None,
    mode="regular",
):
    instances = []
    data_path, gold_path = _resolve_data_paths(
        base_path, dataset, split_num, modal, split
    )
    if data_path is None:
        logging.warning(
            f"Data file not found for modal={modal} split={split} under {base_path}"
        )
        return []

    with open(data_path, encoding="UTF-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            word_idx, tokens = parts
            instances.append(
                {"tokens": tokens.split(), "idx": int(word_idx), "class": None}
            )

    word_classes = get_word_classes(word, modal)
    if gold_path and os.path.exists(gold_path):
        with open(gold_path, encoding="UTF-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx < len(instances):
                    line_class = line.strip()
                    if line_class in word_classes:
                        instances[line_idx]["class"] = word_classes[line_class]

    instances = [inst for inst in instances if inst["class"] is not None]

    if mode == "mfs":
        instances = [i for i in instances if i["class"] == word_classes["1"]]
    elif mode == "lfs":
        instances = [i for i in instances if i["class"] == word_classes["0"]]

    logging.info(
        f"Loaded {len(instances)} instances for {word} {split} from {data_path}"
    )
    return instances