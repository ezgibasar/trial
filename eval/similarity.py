import os
import json
import logging
from collections import defaultdict

import numpy as np

from .data import load_instances, load_coarse_senses
from .vsm import VSM


def create_sense_vectors(
    encoder, model_name, modal, base_path, dataset, split_num, results_path, config
):
    sense_vecs = {}
    ambiguous_words = ["may"] if modal == "may" else ["must"]

    for word in ambiguous_words:
        instances = load_instances(
            word,
            "train",
            modal,
            base_path,
            dataset=dataset,
            split_num=split_num,
            mode=config.mode,
        )
        for inst in instances:
            if encoder.get_num_subtokens(inst["tokens"]) >= config.max_seq_len:
                continue
            try:
                inst_vecs = encoder.token_embeddings(inst["tokens"])[0]
                if inst["idx"] < len(inst_vecs):
                    word_vec = inst_vecs[inst["idx"]][1]
                    word_cls = inst["class"]
                    if word_cls not in sense_vecs:
                        sense_vecs[word_cls] = {
                            "vecs_sum": word_vec,
                            "vecs_num": 1,
                        }
                    else:
                        sense_vecs[word_cls]["vecs_sum"] += word_vec
                        sense_vecs[word_cls]["vecs_num"] += 1
            except Exception:
                continue

    model_safe = model_name.replace("/", "_")
    vectors_dir = os.path.join(
        results_path, "vectors", model_safe, modal, dataset, str(split_num)
    )
    os.makedirs(vectors_dir, exist_ok=True)
    out_path = os.path.join(vectors_dir, "sense_vectors.txt")

    with open(out_path, "w") as f:
        for sense, info in sense_vecs.items():
            vec = info["vecs_sum"] / info["vecs_num"]
            vec_str = " ".join([str(round(v, 6)) for v in vec.tolist()])
            f.write(f"{sense} {vec_str}\n")

    logging.info(f"Saved sense vectors to {out_path}")
    return out_path


def evaluate_similarity(
    encoder,
    sv_path,
    model_name,
    modal,
    base_path,
    dataset,
    split_num,
    results_path,
    config,
):
    senses_vsm = VSM(sv_path, normalize=True)
    all_sense_preds = defaultdict(list)
    all_results = defaultdict(list)

    ambiguous_words = ["may"] if modal == "may" else ["must"]
    coarse_senses = load_coarse_senses(modal)

    for amb_word in ambiguous_words:
        test_instances = load_instances(
            amb_word,
            "test",
            modal,
            base_path,
            dataset=dataset,
            split_num=split_num,
            mode=config.mode,
        )
        for test_inst in test_instances:
            gold_sense = test_inst["class"]
            if encoder.get_num_subtokens(test_inst["tokens"]) >= config.max_seq_len:
                preds = [("NULL", -1)]
            else:
                try:
                    inst_vecs = encoder.token_embeddings(test_inst["tokens"])[0]
                    if test_inst["idx"] >= len(inst_vecs):
                        preds = [("NULL", -1)]
                    else:
                        amb_word_vec = inst_vecs[test_inst["idx"]][1]
                        norm = np.linalg.norm(amb_word_vec)
                        if norm > 0:
                            amb_word_vec = amb_word_vec / norm
                        preds = senses_vsm.most_similar_vec(amb_word_vec, topn=None)
                        preds = [
                            (s, sc) for s, sc in preds if s.split("_")[0] == amb_word
                        ]
                        if not preds:
                            preds = [("NULL", -1)]
                except Exception:
                    preds = [("NULL", -1)]

            all_sense_preds[gold_sense].append(preds)
            all_results[amb_word].append((test_inst, preds))

    all_senses_accs = {}
    all_words_accs = {}
    for amb_word in coarse_senses:
        n_word_correct, n_word_insts = 0, 0
        for sense in coarse_senses[amb_word]:
            sense_preds = all_sense_preds[sense]
            if not sense_preds:
                continue
            n_correct = sum(1 for preds in sense_preds if preds[0][0] == sense)
            all_senses_accs[sense] = n_correct / len(sense_preds)
            n_word_correct += n_correct
            n_word_insts += len(sense_preds)
        if n_word_insts > 0:
            all_words_accs[amb_word] = n_word_correct / n_word_insts

    model_safe = model_name.replace("/", "_")
    results_dir = os.path.join(
        results_path, model_safe, modal, dataset, str(split_num)
    )
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "summary.csv"), "w") as f:
        f.write("word,sense,n_insts,acc\n")
        for amb_word in coarse_senses:
            n_word_insts = 0
            for sense in coarse_senses[amb_word]:
                if sense not in all_senses_accs:
                    continue
                n_sense_insts = len(all_sense_preds[sense])
                n_word_insts += n_sense_insts
                f.write(
                    f"{amb_word},{sense},{n_sense_insts},{all_senses_accs[sense]:.6f}\n"
                )
            if amb_word in all_words_accs:
                f.write(
                    f"{amb_word},ALL,{n_word_insts},{all_words_accs[amb_word]:.6f}\n"
                )

    for amb_word in all_results:
        word_results_path = os.path.join(results_dir, f"{amb_word}.jsonl")
        with open(word_results_path, "w") as f:
            for inst_idx, (test_inst, matches) in enumerate(all_results[amb_word]):
                row = {
                    "idx": inst_idx,
                    "matches": matches,
                    "gold": test_inst["class"],
                    "tokens": test_inst["tokens"],
                }
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    return all_words_accs, all_senses_accs