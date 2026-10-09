# Manifold Embedding: A Point-to-Hyperplane Approach

**Xianglong Shi, Yunhan Jiang, Nicu Sebe, and Ziheng Chen**

## Abstract

Neural networks over non-Euclidean manifolds have attracted increasing attention across various applications. A fundamental step in these networks is to map Euclidean features onto a target manifold. Existing methods mainly use exponential maps and projections, operate through tangent or ambient Euclidean spaces, and may therefore distort intrinsic manifold geometry. On the other hand, point-to-hyperplane distances have recently shown great success in building manifold-valued networks. Based on this, we propose the Point-to-Hyperplane Distance Map (PHDM), a parameter-free mapping that extends Euclidean fully connected layers to Euclidean-to-manifold transformations. Specifically, the exponential map is recovered as a pseudo-distance-based special case. We manifest PHDM on the Lorentz, Poincare, spherical, and projected-spherical models, covering both negative and positive curvature. Experiments on arithmetic reasoning across four LLMs, image classification, graph node classification, and link prediction demonstrate consistent improvements over the corresponding baselines.

## Experiments

| Table | Experiment | Configuration | PHDM implementation |
| --- | --- | --- | --- |
| 4 | Image classification | [paper.py](image/experiments/paper.py), [model configs](image/code/classification/config/) | [classifier.py](image/code/classification/models/classifier.py) |
| 5 | Hypformer | [paper.py](hypformer/experiments/paper.py) | [hyp_layer.py](hypformer/medium/manifolds/hyp_layer.py) |
| 6 | RResNet | [run_table6_grid.py](rresnet/scripts/run_table6_grid.py) | [encoders.py](rresnet/hgcn/models/encoders.py) |
| 7 | LoRA | [table3.py](lora/experiments/table3.py) | [layer.py](lora/peft/tuners/lora/layer.py) |

Mapping classes: `PHDM_Poincare`, `PHDM_Lorentz`, `PHDM_Sphere`, and `PHDM_ProjectSphere`.

## Setup

Use Python 3.10 and CUDA-enabled NVIDIA GPUs.

```bash
git clone https://github.com/Longchentong/PHDM.git
cd PHDM
```

Run the commands below from this directory. Each experiment uses its own environment. Add `--dry-run` to a launcher to preview commands, or `--help` to list options.

## Table 4: Image Classification

CIFAR-10/100 download automatically. Prepare Tiny-ImageNet before training:

```bash
python3.10 -m venv .venv-image
source .venv-image/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r image/requirements.txt

bash image/code/classification/get_tinyimagenet.sh
python image/code/classification/org_tinyimagenet.py
python image/experiments/run_paper.py --gpus 0,1,2,3 --keep-checkpoints
```

Select one configuration with `--datasets CIFAR-10 --geometries lorentz --gpus 0`. Outputs: `image/results/runs/`.

## Table 5: Hypformer

Cora, Citeseer, and PubMed download automatically. Download Airport and run:

```bash
python3.10 -m venv .venv-hypformer
source .venv-hypformer/bin/activate
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 --only-binary=:all: -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
python -m pip install -r hypformer/requirements.txt

python hypformer/scripts/download_airport.py
python hypformer/experiments/run_paper.py --gpus 0,1,2,3
```

Select one dataset with `--datasets airport --gpus 0`. Outputs: `hypformer/results/runs/`.

## Table 6: RResNet

Cora data is included in `rresnet/hgcn/data/cora/`.

```bash
python3.10 -m venv .venv-rresnet
source .venv-rresnet/bin/activate
python -m pip install -r rresnet/requirements.txt
python rresnet/scripts/run_table6_grid.py --gpus 0,1,2,3
```

Select one geometry with `--geometries lorentz --gpus 0`. Outputs: `rresnet/results/table6/`.

## Table 7: LoRA

The commands below use four GPUs with 64 GB of memory each.

```bash
python3.10 -m venv .venv-lora
source .venv-lora/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r lora/requirements.txt

export HF_HOME="$PWD/lora/hf_cache"
hf auth login
for PHDM_MODEL in llama13 gemma7 llama3 qwen25; do
  python lora/experiments/run_table3.py --model "$PHDM_MODEL" --gpus 0,1,2,3 --hf-cache "$HF_HOME"
done
```

Each command trains and evaluates one model. Use `--phase train` or `--phase eval` to run either stage separately.

Adapters: `lora/results/runs/trained_models/`. Predictions: `lora/results/runs/experiment/`.

## Acknowledgements

We thank the authors of:

- [HyperbolicCV](https://github.com/kschwethelm/HyperbolicCV)
- [Hypformer](https://github.com/Graph-and-Geometric-Learning/hyperbolic-transformer)
- [Riemannian Residual Neural Networks](https://github.com/CUAI/Riemannian-Residual-Neural-Networks)
- [HypLoRA](https://github.com/marlin-codes/HypLoRA)
