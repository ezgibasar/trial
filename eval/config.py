MODEL_LIST = [
    "google-bert/bert-base-uncased",
    "FacebookAI/roberta-base",
    "fpadovani/en_child_mlm_13_new",
    "fpadovani/en_wiki_mlm_13_new",
    "EleutherAI/pythia-6.9b",
    "openai-community/gpt2",
    "fpadovani/en_child_clm_13_new",
    "fpadovani/en_wiki_clm_13_new",
]

MODAL_MAP = {
    "may": {"0": "may_root", "1": "may_epistemic"},
    "must": {"0": "must_root", "1": "must_epistemic"},
}

COARSE_SENSES = {
    "may": ["may_root", "may_epistemic"],
    "must": ["must_root", "must_epistemic"],
}


class Config:
    def __init__(
        self,
        data_path=None,
        results_path=None,
        max_seq_len=512,
        layers=None,
        layer_op="sum",
        subword_op="mean",
        mode="regular",
        seed=199,
        device=None,
    ):
        self.data_path = data_path
        self.results_path = results_path
        self.max_seq_len = max_seq_len
        self.layers = layers if layers is not None else [-1, -2, -3, -4]
        self.layer_op = layer_op
        self.subword_op = subword_op
        self.mode = mode
        self.seed = seed
        self.device = device

    def resolve_device(self):
        import torch as th

        if self.device is not None:
            return self.device
        return "cuda" if th.cuda.is_available() else "cpu"