from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"

DATASETS = ("CIFAR-10", "CIFAR-100", "Tiny-ImageNet")
GEOMETRIES = ("poincare", "lorentz")
METHODS = ("phdm",)

# Tiny's historical summary labels were swapped; use the decoder geometry.
BEST_SEEDS = {
    ("poincare", "CIFAR-10"): 1,
    ("poincare", "CIFAR-100"): 1,
    ("poincare", "Tiny-ImageNet"): 1,
    ("lorentz", "CIFAR-10"): 1,
    ("lorentz", "CIFAR-100"): 1,
    ("lorentz", "Tiny-ImageNet"): 777777,
}

CLIP_FEATURES = {
    ("phdm", "poincare", "CIFAR-10"): 2.0,
    ("phdm", "poincare", "CIFAR-100"): 2.0,
    ("phdm", "poincare", "Tiny-ImageNet"): 2.0,
    ("phdm", "lorentz", "CIFAR-10"): 2.0,
    ("phdm", "lorentz", "CIFAR-100"): 1.0,
    ("phdm", "lorentz", "Tiny-ImageNet"): 2.0,
}

def slug(value):
    return value.lower().replace("-", "_").replace(" ", "_")


CONFIGS = {
    (geometry, dataset): CODE_DIR / "classification" / "config" / f"phdm_{geometry}_{slug(dataset)}.txt"
    for geometry in GEOMETRIES
    for dataset in DATASETS
}


@dataclass(frozen=True)
class Job:
    method: str
    geometry: str
    dataset: str
    seed: int

    @property
    def mapping(self):
        return "projector"

    @property
    def clip_features(self):
        return CLIP_FEATURES[(self.method, self.geometry, self.dataset)]

    @property
    def config(self):
        return CONFIGS[(self.geometry, self.dataset)]

    @property
    def run_id(self):
        return "_".join(
            (self.method, self.geometry, slug(self.dataset), f"seed{self.seed}")
        )


def build_jobs(methods=METHODS, geometries=GEOMETRIES, datasets=DATASETS, seeds=None):
    return [
        Job(method, geometry, dataset, seed)
        for geometry in geometries
        for dataset in datasets
        for seed in ((BEST_SEEDS[(geometry, dataset)],) if seeds is None else seeds)
        for method in methods
    ]
