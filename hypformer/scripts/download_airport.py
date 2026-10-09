#!/usr/bin/env python3
import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://raw.githubusercontent.com/Graph-and-Geometric-Learning/"
    "hyperbolic-transformer/768fcec099bcb039d0d838e3e4ebe83ef6bb7426/"
    "data/hgcn_data/airport/airport.p"
)
SHA256 = "ab16b04cea3666e87dbd8a5bff3de68da61330f4604fa21082f7a17e806e64d8"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Download the pinned Hypformer Airport data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "hgcn_data" / "airport" / "airport.p",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and digest(output) == SHA256 and not args.force:
        print(output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    urllib.request.urlretrieve(URL, temporary)
    actual = digest(temporary)
    if actual != SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch: expected {SHA256}, received {actual}")
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
