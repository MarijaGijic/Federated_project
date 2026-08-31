"""
Reassemble split client-data archives into data/client{N}/ directories.

Each client's prepared data (images/ + annotations.csv) was archived as a
tar.gz and, for larger clients, split into raw byte chunks named
client{N}_{PART}, e.g.:

    client1_01                       (single part)
    client2_01  client2_02  client2_03
    client3_01

This script groups parts by client number, concatenates them in order,
and extracts the result under --output_dir (each archive contains its own
"data/client{N}/" prefix, so extraction lands at output_dir/data/client{N}).

Usage (Colab or local, requires `cat` and `tar` on PATH):
    python concat_client_archives.py \
        --source_dir /content/drive/MyDrive/mammography_parts \
        --output_dir /content
"""

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

PART_RE = re.compile(r"^client(\d+)_(\d+)$")


def group_parts(source_dir: Path) -> dict:
    groups = defaultdict(list)
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        m = PART_RE.match(path.name)
        if m:
            client_id, part_num = int(m.group(1)), int(m.group(2))
            groups[client_id].append((part_num, path))
    for client_id in groups:
        groups[client_id].sort(key=lambda t: t[0])
    return groups


def reassemble_and_extract(client_id: int, parts: list, output_dir: Path):
    part_paths = [str(p) for _, p in parts]
    print(f"[client{client_id}] {len(part_paths)} part(s): {[p.name for _, p in parts]}")

    cat = subprocess.Popen(["cat", *part_paths], stdout=subprocess.PIPE)
    tar = subprocess.Popen(["tar", "-xz", "-C", str(output_dir)], stdin=cat.stdout)
    cat.stdout.close()
    tar.communicate()
    cat.wait()

    if cat.returncode != 0:
        raise RuntimeError(f"[client{client_id}] cat failed with exit code {cat.returncode}")
    if tar.returncode != 0:
        raise RuntimeError(
            f"[client{client_id}] tar extraction failed (exit code {tar.returncode}) "
            f"- parts may be incomplete or corrupted"
        )

    print(f"[client{client_id}] extracted -> {output_dir}/data/client{client_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_dir", required=True, help="Directory containing client{N}_{PART} files")
    parser.add_argument("--output_dir", required=True, help="Directory to extract into")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_parts(source_dir)
    if not groups:
        raise SystemExit(f"No client{{N}}_{{PART}} files found in {source_dir}")

    for client_id in sorted(groups):
        reassemble_and_extract(client_id, groups[client_id], output_dir)

    print("\nDone. Client directories:")
    for client_id in sorted(groups):
        print(f"  {output_dir}/data/client{client_id}")


if __name__ == "__main__":
    main()
