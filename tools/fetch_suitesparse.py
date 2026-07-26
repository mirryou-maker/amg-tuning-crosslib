#!/usr/bin/env python3
"""Download SuiteSparse Matrix Collection matrices for the Phase 1 pilot.

Uses the official index (ssstats.csv) to filter candidates, then fetches
Matrix Market tarballs. Downloads are cached: re-running skips existing files.

Usage:
    python fetch_suitesparse.py --list-only          # show candidates, download nothing
    python fetch_suitesparse.py --manifest sel.csv   # download matrices named in a manifest
"""

import argparse
import csv
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"
MM_URL = "https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "matrices"
INDEX_CACHE = ROOT / "data" / "ssstats.csv"


def fetch(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "phase1-pilot/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def load_index():
    """Return list of dicts describing every matrix in the collection.

    ssstats.csv has two header lines (count, timestamp) then one row per matrix:
    group, name, nrows, ncols, nnz, is_real, is_binary, is_2d3d, is_spd, pattern_symmetry, ...
    """
    if not INDEX_CACHE.exists():
        INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        INDEX_CACHE.write_bytes(fetch(INDEX_URL))
    rows = list(csv.reader(io.StringIO(INDEX_CACHE.read_text())))
    out = []
    for r in rows[2:]:
        if len(r) < 10:
            continue
        try:
            out.append({
                "group": r[0], "name": r[1],
                "nrows": int(r[2]), "ncols": int(r[3]), "nnz": int(r[4]),
                "real": r[5] == "1", "binary": r[6] == "1",
                "is2d3d": r[7] == "1", "spd": r[8] == "1",
                "psym": float(r[9]),
            })
        except ValueError:
            continue
    return out


def candidates(index, nmin, nmax, spd_only):
    """Square, real, non-binary matrices in the pilot size window."""
    sel = []
    for m in index:
        if m["nrows"] != m["ncols"]:
            continue
        if not m["real"] or m["binary"]:
            continue
        if not (nmin <= m["nrows"] <= nmax):
            continue
        if spd_only and not m["spd"]:
            continue
        sel.append(m)
    return sel


def download(group, name, force=False):
    """Fetch and extract one matrix; return path to its .mtx file."""
    dest = CACHE / group / name
    if dest.exists() and not force:
        mtx = sorted(dest.glob("*.mtx"))
        if mtx:
            return mtx[0]
    dest.mkdir(parents=True, exist_ok=True)
    blob = fetch(MM_URL.format(group=group, name=name))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            # flatten: drop the leading <name>/ directory from the archive
            member.name = Path(member.name).name
            tf.extract(member, dest)
    mtx = sorted(dest.glob("*.mtx"))
    if not mtx:
        raise RuntimeError(f"no .mtx found in {group}/{name}")
    return mtx[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmin", type=int, default=10_000)
    ap.add_argument("--nmax", type=int, default=500_000)
    ap.add_argument("--spd-only", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--manifest", type=Path,
                    help="CSV with group,name columns; download exactly these")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.manifest:
        rows = list(csv.DictReader(args.manifest.open()))
        for i, r in enumerate(rows, 1):
            try:
                p = download(r["group"], r["name"], force=args.force)
                print(f"[{i}/{len(rows)}] {r['group']}/{r['name']} -> {p}")
            except Exception as e:
                print(f"[{i}/{len(rows)}] {r['group']}/{r['name']} FAILED: {e}",
                      file=sys.stderr)
        return

    index = load_index()
    sel = candidates(index, args.nmin, args.nmax, args.spd_only)
    print(f"collection: {len(index)} matrices; candidates: {len(sel)}")
    if args.list_only:
        w = csv.writer(sys.stdout)
        w.writerow(["group", "name", "nrows", "nnz", "spd", "psym", "is2d3d"])
        for m in sorted(sel, key=lambda m: m["nrows"]):
            w.writerow([m["group"], m["name"], m["nrows"], m["nnz"],
                        int(m["spd"]), m["psym"], int(m["is2d3d"])])


if __name__ == "__main__":
    main()
