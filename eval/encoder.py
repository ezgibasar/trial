import logging
import numpy as np
import torch as th
from transformers import AutoModel, AutoTokenizer, set_seed


class TransformerEncoder:
    def __init__(self, model_name_or_path, config):
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.nlm_model = None
        self.nlm_tokenizer = None
        self.device = config.resolve_device()
        self.remove_cls_sep = False
        self._load()

    def _load(self):
        set_seed(self.config.seed)
        logging.info(f"Loading model: {self.model_name_or_path} on {self.device}")

        self.nlm_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        if self.nlm_tokenizer.pad_token is None:
            self.nlm_tokenizer.pad_token = (
                self.nlm_tokenizer.eos_token
                if self.nlm_tokenizer.eos_token
                else "[PAD]"
            )

        self.nlm_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            output_hidden_states=True,
            trust_remote_code=True,
        )

        model_type = self.nlm_model.config.model_type.lower()
        if "bert" in model_type or "roberta" in model_type:
            self.remove_cls_sep = True

        self.nlm_model.eval()
        self.nlm_model.to(self.device)

    def get_encodings(self, sentence):
        list_word_encodings = []
        for word_id in range(len(sentence)):
            if word_id == 0:
                word_temp = self.nlm_tokenizer(
                    sentence[word_id],
                    add_special_tokens=False,
                    padding=False,
                    return_attention_mask=False,
                    truncation=True,
                )
            else:
                word_temp = self.nlm_tokenizer(
                    " " + sentence[word_id],
                    add_special_tokens=False,
                    padding=False,
                    return_attention_mask=False,
                    truncation=True,
                )
            list_word_encodings.append(word_temp["input_ids"])
        return [list_word_encodings]

    def merge_subword_embeddings(
        self, tokens, encodings, embeddings, return_tokens=True
    ):
        tok_embeddings = []
        encoding_idx = 0

        for tok, tok_encodings in zip(tokens, encodings):
            if self.config.subword_op == "mean":
                tok_embedding = th.zeros(embeddings.shape[-1]).to(self.device)
                num_used = 0
                for _ in tok_encodings:
                    if encoding_idx < len(embeddings):
                        tok_embedding += embeddings[encoding_idx]
                        encoding_idx += 1
                        num_used += 1
                if num_used > 0:
                    tok_embedding = tok_embedding / num_used
            elif self.config.subword_op == "first":
                tok_embedding = (
                    embeddings[encoding_idx]
                    if encoding_idx < len(embeddings)
                    else th.zeros(embeddings.shape[-1]).to(self.device)
                )
                encoding_idx += len(tok_encodings)
            else:
                raise ValueError(f"Invalid subword_op: {self.config.subword_op}")

            tok_embedding = tok_embedding.detach().cpu().numpy()
            if return_tokens:
                tok_embeddings.append((tok, tok_embedding))
            else:
                tok_embeddings.append(tok_embedding)

        return tok_embeddings

    def get_num_subtokens(self, tokens):
        try:
            return len(self.get_encodings(tokens)[0])
        except Exception:
            return len(tokens)

    def token_embeddings(self, batch_sent_tokens, return_tokens=True):
        batch_sent_encodings = self.get_encodings(batch_sent_tokens)
        inst = self.nlm_tokenizer(
            " ".join(batch_sent_tokens),
            add_special_tokens=True,
            padding=False,
            return_attention_mask=True,
            truncation=True,
            max_length=self.config.max_seq_len,
        )

        input_ids = th.tensor([inst["input_ids"]]).to(self.device)
        input_mask = th.tensor([inst["attention_mask"]]).to(self.device)

        with th.no_grad():
            outputs = self.nlm_model(input_ids, attention_mask=input_mask)
            batch_hidden_states = outputs["hidden_states"]

        num_layers = len(batch_hidden_states)
        selected_layers = [
            l if l >= 0 else num_layers + l for l in self.config.layers
        ]
        sel_hidden_states = [batch_hidden_states[i] for i in selected_layers]

        merged_batch_hidden_states = []
        for layer_hidden_states in sel_hidden_states:
            merged_layer_hidden_states = []
            for sent_idx, sent_embeddings in enumerate(layer_hidden_states):
                if self.remove_cls_sep and len(sent_embeddings) > 2:
                    sent_embeddings = sent_embeddings[1:-1]
                sent_encodings = batch_sent_encodings[sent_idx]
                sent_embeddings = self.merge_subword_embeddings(
                    batch_sent_tokens,
                    sent_encodings,
                    sent_embeddings,
                    return_tokens=return_tokens,
                )
                merged_layer_hidden_states.append(sent_embeddings)
            merged_batch_hidden_states.append(merged_layer_hidden_states)

        combined_batch_embeddings = []
        for sent_idx, sent_tokens in enumerate([batch_sent_tokens]):
            combined_sent_embeddings = []
            for tok_idx in range(len(sent_tokens)):
                tok_layer_vecs = []
                for layer_idx in range(len(merged_batch_hidden_states)):
                    if (
                        sent_idx < len(merged_batch_hidden_states[layer_idx])
                        and tok_idx
                        < len(merged_batch_hidden_states[layer_idx][sent_idx])
                    ):
                        tok_layer_vecs.append(
                            merged_batch_hidden_states[layer_idx][sent_idx][tok_idx][
                                1
                            ]
                        )
                if len(tok_layer_vecs) == 0:
                    continue
                elif len(tok_layer_vecs) == 1:
                    tok_combined_vec = tok_layer_vecs[0]
                else:
                    stacked = np.array(tok_layer_vecs)
                    if self.config.layer_op == "sum":
                        tok_combined_vec = stacked.sum(axis=0)
                    else:
                        tok_combined_vec = stacked.mean(axis=0)
                tok = merged_batch_hidden_states[-1][sent_idx][tok_idx][0]
                combined_sent_embeddings.append((tok, tok_combined_vec))
            combined_batch_embeddings.append(combined_sent_embeddings)

        return combined_batch_embeddings