import os
import json
import logging

import torch as th

from .data import load_instances


def _is_causal(model):
    model_type = model.config.model_type.lower()
    return "gpt" in model_type or "neox" in model_type


def _token_positions(tokenizer, sentence, tokens, target_word_idx):
    offsets = tokenizer(
        sentence,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )["offset_mapping"][0].tolist()

    char_start = sum(len(t) + 1 for t in tokens[:target_word_idx])
    char_end = char_start + len(tokens[target_word_idx])

    return [
        idx
        for idx, (s, e) in enumerate(offsets)
        if not (e <= char_start or s >= char_end)
    ]


def _word_surprisal(model, tokenizer, tokens, target_word_idx, device, causal):
    sentence = " ".join(tokens)
    enc = tokenizer(
        sentence, return_tensors="pt", truncation=True, max_length=512
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    target_ids = _token_positions(tokenizer, sentence, tokens, target_word_idx)
    if not target_ids:
        return None

    with th.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    log_probs = th.log_softmax(logits[0], dim=-1)
    total = 0.0
    count = 0
    for tid in target_ids:
        if tid == 0:
            continue
        token_id = input_ids[0, tid].item()
        total += -log_probs[tid - 1, token_id].item()
        count += 1

    return total / count if count > 0 else None


def evaluate_surprisal(
    model,
    tokenizer,
    model_name,
    modal,
    base_path,
    dataset,
    split_num,
    results_path,
    config,
):
    device = config.resolve_device()
    causal = _is_causal(model)

    ambiguous_words = ["may"] if modal == "may" else ["must"]
    all_results = {}

    for amb_word in ambiguous_words:
        instances = load_instances(
            amb_word,
            "test",
            modal,
            base_path,
            dataset=dataset,
            split_num=split_num,
            mode=config.mode,
        )
        results = []
        for inst in instances:
            try:
                score = _word_surprisal(
                    model,
                    tokenizer,
                    inst["tokens"],
                    inst["idx"],
                    device,
                    causal,
                )
            except Exception:
                score = None
            results.append(
                {
                    "tokens": inst["tokens"],
                    "idx": inst["idx"],
                    "gold": inst["class"],
                    "surprisal": score,
                }
            )
        all_results[amb_word] = results

    model_safe = model_name.replace("/", "_")
    results_dir = os.path.join(
        results_path, model_safe, modal, dataset, str(split_num)
    )
    os.makedirs(results_dir, exist_ok=True)

    summary = {}
    for amb_word, rows in all_results.items():
        by_class = {}
        for r in rows:
            if r["surprisal"] is None:
                continue
            by_class.setdefault(r["gold"], []).append(r["surprisal"])
        summary[amb_word] = {
            sense: {
                "n": len(vals),
                "mean_surprisal": sum(vals) / len(vals),
            }
            for sense, vals in by_class.items()
        }

        with open(
            os.path.join(results_dir, f"{amb_word}_surprisal.jsonl"), "w"
        ) as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(results_dir, "surprisal_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary