#!/usr/bin/env bash
set -euo pipefail
TINY_DATA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data
TINY_DATA_URL=https://cs231n.stanford.edu/tiny-imagenet-200.zip
mkdir -p "$TINY_DATA_DIR"
if [[ -f "$TINY_DATA_DIR/tiny-imagenet-200/val/val_annotations.txt" ]]; then
  printf '%s\n' "$TINY_DATA_DIR/tiny-imagenet-200"
  exit 0
fi
TINY_ARCHIVE=$TINY_DATA_DIR/tiny-imagenet-200.zip
if command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 "$TINY_DATA_URL" --output "$TINY_ARCHIVE.part"
else
  wget "$TINY_DATA_URL" -O "$TINY_ARCHIVE.part"
fi
mv "$TINY_ARCHIVE.part" "$TINY_ARCHIVE"
unzip -q "$TINY_ARCHIVE" -d "$TINY_DATA_DIR"
rm "$TINY_ARCHIVE"
printf '%s\n' "$TINY_DATA_DIR/tiny-imagenet-200"
