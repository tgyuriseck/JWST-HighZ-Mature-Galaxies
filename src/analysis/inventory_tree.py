# C:\JWST-Mature-Galaxies\src\analysis\inventory_tree_v2.py
# Inventory your project folder and save a tree, a file list (CSV), and a summary.
# Runs from VS Code with you already in: C:\JWST-Mature-Galaxies\src>

from __future__ import annotations
import argparse
import csv
import datetime as dt
import os
from pathlib import Path
from typing import Iterable, List, Tuple

# ---------- Helpers ----------

def human_bytes(n: int) -> str:
    """Convert bytes to a short human-readable string."""
    step = 1024.0
    if n < step: return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= step
        if n < step:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"

def is_ignored(path: Path, ignore_globs: List[str]) -> bool:
    """Return True if the path matches any ignore pattern (case-insensitive)."""
    p = str(path).lower().replace("\\", "/")
    for pat in ignore_globs:
        if path.match(pat) or Path(p).match(pat.lower()):
            return True
        # also allow substring-style ignore (e.g. "/.git/")
        if pat.strip().lower() in p:
            return True
    return False

def shorten(s: str, maxlen: int = 120) -> str:
    return s if len(s) <= maxlen else s[:maxlen-3] + "..."

# ---------- Core logic ----------

def walk_tree(
    root: Path,
    max_depth: int = 50,
    ignore_globs: List[str] | None = None,
) -> Tuple[List[Tuple[int, Path, int]], List[Tuple[Path, int, float]]]:
    """
    Walk the tree yielding (depth, path, size_bytes) for directories,
    and (file_path, size_bytes, mtime) for files.
    Depth 0 corresponds to the root itself.
    """
    if ignore_globs is None:
        ignore_globs = []

    dir_rows: List[Tuple[int, Path, int]] = []
    file_rows: List[Tuple[Path, int, float]] = []

    # Pre-compute all entries with a manual stack so we can track depth
    stack: List[Tuple[int, Path]] = [(0, root)]

    while stack:
        depth, current = stack.pop()
        if depth > max_depth:
            continue
        if is_ignored(current, ignore_globs):
            continue

        if current.is_dir():
            # Directory size we'll compute as sum of immediate files (not recursive, keeps it quick)
            try:
                entries = list(current.iterdir())
            except PermissionError:
                entries = []
            size_here = 0
            for e in entries:
                if e.is_file():
                    try:
                        size_here += e.stat().st_size
                    except OSError:
                        pass
            dir_rows.append((depth, current, size_here))

            # Push children onto stack (reverse sort so printed order is alphabetical)
            dirs = sorted([e for e in entries if e.is_dir()], key=lambda p: p.name.lower(), reverse=True)
            files = sorted([e for e in entries if e.is_file()], key=lambda p: p.name.lower())

            # record files
            for f in files:
                if is_ignored(f, ignore_globs):
                    continue
                try:
                    st = f.stat()
                    file_rows.append((f, st.st_size, st.st_mtime))
                except OSError:
                    file_rows.append((f, 0, 0.0))

            # schedule dirs
            for d in dirs:
                if is_ignored(d, ignore_globs):
                    continue
                stack.append((depth + 1, d))

        else:
            # If root is a file (rare), record it
            try:
                st = current.stat()
                file_rows.append((current, st.st_size, st.st_mtime))
            except OSError:
                file_rows.append((current, 0, 0.0))

    # Sort directories by their path for consistent output
    dir_rows.sort(key=lambda t: str(t[1]).lower())
    # Files are fine as collected; we will sort when exporting CSV
    return dir_rows, file_rows

def write_tree_txt(
    out_path: Path,
    dir_rows: List[Tuple[int, Path, int]],
    file_rows: List[Tuple[Path, int, float]],
    root: Path,
) -> None:
    """
    Write a pretty tree view with sizes.
    """
    # Build a quick lookup from directory to its files (immediate children only)
    from collections import defaultdict
    files_by_dir = defaultdict(list)
    for f, sz, mtime in file_rows:
        parent = f.parent
        files_by_dir[parent].append((f, sz, mtime))

    # We need directories in hierarchical order; dir_rows is path-sorted already.
    with out_path.open("w", encoding="utf-8") as fp:
        fp.write(f"Project tree: {root}\n")
        fp.write("=" * (len(str(root)) + 14) + "\n\n")
        # Print each directory and its immediate files
        for depth, d, d_size in dir_rows:
            indent = "  " * depth
            fp.write(f"{indent}{d.name}/  [files size here: {human_bytes(d_size)}]\n")
            # list files
            for f, sz, _ in sorted(files_by_dir.get(d, []), key=lambda t: t[0].name.lower()):
                fp.write(f"{indent}  {f.name}  ({human_bytes(sz)})\n")
        fp.write("\n")

def write_files_csv(out_csv: Path, file_rows: List[Tuple[Path, int, float]], root: Path) -> None:
    """
    Write full file listing as CSV with columns: rel_path, abs_path, size_bytes, size_human, mtime_iso
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "abs_path", "size_bytes", "size_human", "mtime_iso"])
        for p, sz, mt in sorted(file_rows, key=lambda t: str(t[0]).lower()):
            rel = os.path.relpath(p, root)
            mtime_iso = dt.datetime.fromtimestamp(mt).isoformat() if mt else ""
            w.writerow([rel, str(p), sz, human_bytes(sz), mtime_iso])

def write_summary_txt(
    out_path: Path,
    dir_rows: List[Tuple[int, Path, int]],
    file_rows: List[Tuple[Path, int, float]],
    root: Path,
    top_n: int = 20,
) -> None:
    total_files = len(file_rows)
    total_dirs = len(dir_rows)
    total_bytes = sum(sz for _, sz, _ in file_rows)

    # largest files
    largest = sorted(file_rows, key=lambda t: t[1], reverse=True)[:top_n]

    # extension counts
    from collections import Counter
    ext_counter = Counter([p.suffix.lower() for p, _, _ in file_rows])
    common_ext = ext_counter.most_common(20)

    with out_path.open("w", encoding="utf-8") as fp:
        fp.write(f"SUMMARY for {root}\n")
        fp.write("=" * (len(str(root)) + 12) + "\n\n")
        fp.write(f"Total directories: {total_dirs}\n")
        fp.write(f"Total files      : {total_files}\n")
        fp.write(f"Total size       : {human_bytes(total_bytes)} ({total_bytes} bytes)\n\n")

        fp.write("Most common extensions (top 20):\n")
        for ext, cnt in common_ext:
            fp.write(f"  {ext or '(no ext)'} : {cnt}\n")
        fp.write("\n")

        fp.write(f"Largest files (top {top_n}):\n")
        for p, sz, mt in largest:
            fp.write(f"  {human_bytes(sz):>10}  {p}\n")
        fp.write("\n")

        # Little hint section for our figure work
        fp.write("Notes for figure pipeline wiring:\n")
        fp.write("  • Look for CSVs with xi(d) columns (e.g., 'bin_right_Mpc', 'xi', 'xi_err').\n")
        fp.write("  • Look for spectra/FFT CSVs with 'wavelength_Mpc' and 'power'.\n")
        fp.write("  • Check for photo-z MC folders containing many xi CSVs per tier.\n")
        fp.write("  • Verify variance/mock outputs for z=8–10 and z=10–20 histograms.\n")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Inventory the project folder and write tree, CSV, and summary.")
    ap.add_argument("--root", default="..", help="Root folder to inventory (default: .. i.e., project root from src)")
    ap.add_argument("--max-depth", type=int, default=50, help="Max directory depth to traverse (default: 50)")
    ap.add_argument(
        "--ignore",
        nargs="*",
        default=[
            # Common ignores; adjust as needed
            "*.git*", "*__pycache__*", "*.ipynb_checkpoints*",
            "*\\.venv*", "*\\env*", "*\\venv*", "*\\node_modules*",
            "*\\__pycache__*", "*\\tmp*", "*\\temp*",
        ],
        help="Glob patterns (or substrings) to ignore (space-separated).",
    )
    ap.add_argument("--top-n", type=int, default=20, help="Top N largest files in summary (default: 20)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Prepare output dir
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (root / "results" / "audit" / stamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Walk
    dir_rows, file_rows = walk_tree(root, max_depth=args.max_depth, ignore_globs=args.ignore)

    # Write outputs
    tree_txt = out_dir / "tree.txt"
    files_csv = out_dir / "files.csv"
    summary_txt = out_dir / "summary.txt"

    write_tree_txt(tree_txt, dir_rows, file_rows, root)
    write_files_csv(files_csv, file_rows, root)
    write_summary_txt(summary_txt, dir_rows, file_rows, root, top_n=args.top_n)

    # Console summary
    print("=== Inventory complete ===")
    print(f"Root         : {root}")
    print(f"Directories  : {len(dir_rows)}")
    print(f"Files        : {len(file_rows)}")
    print(f"Outputs:")
    print(f"  - {tree_txt}")
    print(f"  - {files_csv}")
    print(f"  - {summary_txt}")

if __name__ == "__main__":
    main()
