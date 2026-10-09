from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "medium"

DATASETS = ("airport", "cora", "citeseer", "pubmed")
METHODS = ("phdm",)
PROTOCOLS = ("paper",)
VALIDATION_POLICIES = ("validation_accuracy", "validation_accuracy_then_loss")
SELECTED_RUNS = {"airport": 4, "cora": 1, "citeseer": 1, "pubmed": 4}

BASELINE_CONFIGS = {
    "airport": {
        "lr": 0.005,
        "weight_decay": 1e-3,
        "hidden_channels": 256,
        "use_graph": 1,
        "gnn_dropout": 0.4,
        "gnn_use_bn": 1,
        "gnn_num_layers": 3,
        "gnn_use_init": 1,
        "trans_num_layers": 1,
        "trans_use_residual": 1,
        "trans_use_bn": 0,
        "graph_weight": 0.2,
        "trans_dropout": 0.2,
        "power_k": 2.0,
        "epochs": 5000,
        "k_in": 1.0,
        "k_out": 2.0,
        "decoder_type": "hyp",
        "attention_type": "linear_focused",
        "seed": 42,
    },
    "cora": {
        "lr": 0.005,
        "hidden_channels": 256,
        "use_graph": 1,
        "gnn_num_layers": 6,
        "graph_weight": 0.5,
        "weight_decay": 0.005,
        "gnn_use_residual": 1,
        "gnn_dropout": 0.5,
        "gnn_use_bn": 0,
        "gnn_use_init": 0,
        "trans_num_layers": 1,
        "trans_dropout": 0.5,
        "trans_use_residual": 1,
        "rand_split_class": 1,
        "no_feat_norm": 1,
        "trans_use_bn": 0,
        "trans_num_heads": 1,
        "valid_num": 500,
        "test_num": 1000,
        "epochs": 500,
        "seed": 123,
        "power_k": 1.0,
        "attention_type": "linear_focused",
        "decoder_type": "euc",
        "k_in": 1.0,
        "k_out": 3.0,
    },
    "citeseer": {
        "lr": 0.005,
        "hidden_channels": 256,
        "use_graph": 1,
        "weight_decay": 0.005,
        "gnn_num_layers": 5,
        "graph_weight": 0.4,
        "gnn_dropout": 0.5,
        "gnn_use_weight": 0,
        "gnn_use_bn": 0,
        "gnn_use_residual": 1,
        "gnn_use_init": 0,
        "gnn_use_act": 1,
        "trans_num_layers": 1,
        "trans_dropout": 0.5,
        "trans_use_residual": 1,
        "trans_use_weight": 1,
        "trans_num_heads": 1,
        "trans_use_bn": 0,
        "trans_use_act": 0,
        "rand_split_class": 1,
        "valid_num": 500,
        "test_num": 1000,
        "no_feat_norm": 1,
        "add_positional_encoding": 1,
        "epochs": 500,
        "seed": 123,
        "power_k": 3.0,
        "k_in": 1.0,
        "k_out": 1.0,
        "attention_type": "linear_focused",
        "decoder_type": "euc",
        "save_result": 0,
    },
    "pubmed": {
        "lr": 0.005,
        "weight_decay": 5e-4,
        "hidden_channels": 256,
        "use_graph": 1,
        "gnn_num_layers": 4,
        "graph_weight": 0.8,
        "gnn_use_residual": 1,
        "gnn_use_weight": 0,
        "gnn_dropout": 0.5,
        "gnn_use_bn": 0,
        "gnn_use_init": 0,
        "gnn_use_act": 0,
        "trans_num_layers": 1,
        "trans_use_weight": 1,
        "trans_use_act": 0,
        "trans_dropout": 0.5,
        "trans_use_residual": 1,
        "rand_split_class": 1,
        "valid_num": 500,
        "test_num": 1000,
        "no_feat_norm": 1,
        "epochs": 500,
        "seed": 123,
        "power_k": 3.0,
        "k_in": 1.0,
        "k_out": 2.0,
        "attention_type": "linear_focused",
        "decoder_type": "hyp",
        "hyp_lr": 0.005,
        "hyp_weight_decay": 5e-4,
        "save_result": 0,
    },
}

PHDM_CONFIGS = {dataset: dict(config, val_tie_break=1) for dataset, config in BASELINE_CONFIGS.items()}
PHDM_CONFIGS["airport"].update(
    weight_decay=2e-3, seed=444444, gnn_dropout=0.2, trans_dropout=0.0, graph_weight=0.5,
    lr_schedule="plateau", lr_decay_patience=25, patience=500,
)
PHDM_CONFIGS["cora"].update(
    graph_weight=0.7, weight_decay=0.032, lr=0.01, epochs=1500,
    patience=750, no_decay_bias_norm=1,
)
PHDM_CONFIGS["citeseer"].update(
    graph_weight=0.3, weight_decay=0.005, gnn_dropout=0.3, trans_dropout=0.3,
    power_k=4.0, k_out=0.25, patience=500,
)
PHDM_CONFIGS["pubmed"].update(
    weight_decay=0.0016, seed=3407, graph_weight=0.95, gnn_dropout=0.6,
    trans_dropout=0.4, epochs=1000, patience=500, power_k=1.0, k_in=0.5,
    hyp_lr=0.01, hyp_weight_decay=0.0005, graph_lr_multiplier=1.0,
    trans_lr_multiplier=1.0, no_decay_bias_norm=0, lr_schedule="none",
    label_smoothing=0.05, stable_focusing=1,
)


@dataclass(frozen=True)
class Job:
    method: str
    dataset: str
    protocol: str = "paper"

    @property
    def input_map(self):
        return "phdm"

    @property
    def run_id(self):
        return f"{self.protocol}_{self.method}_{self.dataset}"

    @property
    def config(self):
        return dict(PHDM_CONFIGS[self.dataset])

    @property
    def selected_run(self):
        return SELECTED_RUNS[self.dataset]


def build_jobs(methods=METHODS, datasets=DATASETS, protocol="paper"):
    return [Job(method, dataset, protocol) for dataset in datasets for method in methods]
