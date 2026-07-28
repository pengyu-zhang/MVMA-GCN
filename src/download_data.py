"""Download the dataset tarball from the GitHub Release mirror and extract it.

The three processed multi-view datasets (ACM, DBLP, IMDB) are distributed as
a tar.gz asset attached to this repository's GitHub Release (they are small,
~16 MB unpacked). See data/README.md for provenance and attribution.

Usage:
    python -m src.download_data [--dest data/raw] [--tarball path/to/local.tar.gz]
"""

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request

DATA_URL = (
    "https://github.com/pengyu-zhang/MVMA-GCN/releases/download/"
    "v1.1/mvma-gcn-data.tar.gz"
)
# sha256 of the release asset; verified after download
DATA_SHA256 = "2ee9de3c9fc712f0209e85e8a1c84276aa044958f3816ccd3c9b760e12a851d2"

EXPECTED_DIRS = ("acm", "DBLP", "IMDB", "BlogCatalog", "flickr", "citeseer", "uai")


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest_path):
    print(f"downloading {url}")

    def report(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            sys.stdout.write(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=report)
    print()


def main():
    parser = argparse.ArgumentParser(description="Download MVMA-GCN datasets")
    parser.add_argument("--dest", default="data/raw")
    parser.add_argument(
        "--tarball",
        default=None,
        help="use an existing local tar.gz instead of downloading",
    )
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    if all(os.path.isdir(os.path.join(args.dest, d)) for d in EXPECTED_DIRS):
        print(f"datasets already present in {args.dest}, nothing to do")
        return

    tarball = args.tarball
    if tarball is None:
        tarball = os.path.join(args.dest, "mvma-gcn-data.tar.gz")
        download(DATA_URL, tarball)
        if len(DATA_SHA256) == 64:
            digest = sha256sum(tarball)
            if digest != DATA_SHA256:
                raise RuntimeError(
                    f"checksum mismatch: expected {DATA_SHA256}, got {digest}"
                )
            print("checksum OK")

    print(f"extracting {tarball} -> {args.dest}")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(args.dest, filter="data")
    missing = [d for d in EXPECTED_DIRS if not os.path.isdir(os.path.join(args.dest, d))]
    if missing:
        raise RuntimeError(f"tarball did not contain expected directories: {missing}")
    print("done")


if __name__ == "__main__":
    main()
