#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$1" == "--help" ]]; then
  printf '%s\n' \
    "Usage: bash scripts/run_table6.sh ARCHITECTURE GEOMETRY [training arguments]" \
    "ARCHITECTURE: rresnet, graph" \
    "GEOMETRY: lorentz, poincare, sphere, projected_sphere" \
    "Pass the original training parameters and seed as hgcn/train.py arguments." \
    "The task, dataset, curvature and PHDM projection are fixed to Table 6."
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi

TABLE6_ARCHITECTURE=$1
TABLE6_GEOMETRY=$2
shift 2

case "$TABLE6_ARCHITECTURE" in
  rresnet) TABLE6_MODEL_PREFIX=RRNet ;;
  graph) TABLE6_MODEL_PREFIX=RRNetGraph ;;
  *) printf 'Unknown architecture: %s\n' "$TABLE6_ARCHITECTURE" >&2; exit 2 ;;
esac

case "$TABLE6_GEOMETRY" in
  lorentz)
    TABLE6_MODEL="${TABLE6_MODEL_PREFIX}HyperbolicLorentz"
    TABLE6_MANIFOLD=Hyperboloid
    TABLE6_PROJECTION=lorentz_fc
    ;;
  poincare)
    TABLE6_MODEL="${TABLE6_MODEL_PREFIX}Hyperbolic"
    TABLE6_MANIFOLD=PoincareBall
    TABLE6_PROJECTION=poincare_fc
    ;;
  sphere)
    TABLE6_MODEL="${TABLE6_MODEL_PREFIX}Spherical"
    TABLE6_MANIFOLD=Sphere
    TABLE6_PROJECTION=sphere_exact
    ;;
  projected_sphere)
    TABLE6_MODEL="${TABLE6_MODEL_PREFIX}ProjectedSpherical"
    TABLE6_MANIFOLD=ProjectedSphere
    TABLE6_PROJECTION=psphere_fc
    ;;
  *) printf 'Unknown geometry: %s\n' "$TABLE6_GEOMETRY" >&2; exit 2 ;;
esac

TABLE6_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HGCN_HOME="$TABLE6_ROOT/hgcn"
export DATAPATH="$TABLE6_ROOT/hgcn/data"
export LOG_DIR="${LOG_DIR:-$TABLE6_ROOT/results/table6}"
export PYTHONPATH="$TABLE6_ROOT/hgcn:$TABLE6_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "${PHDM_PYTHON:-python}" "$TABLE6_ROOT/hgcn/train.py" "$@" \
  --task lp --dataset cora --c 1.0 \
  --model "$TABLE6_MODEL" --manifold "$TABLE6_MANIFOLD" \
  --proj-type "$TABLE6_PROJECTION"
