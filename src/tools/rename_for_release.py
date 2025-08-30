#!/usr/bin/env python3
"""
rename_for_release.py — Clean reviewer-facing names by removing *_vN[a]? suffixes
from filenames and updating imports / string literals accordingly.

Usage (from VS Code terminal at PS C:\JWST-HighZ-Mature-Galaxies\src>):
  python tools\rename_for_release.py --dry-run   # show what will change
  python tools\rename_for_release.py --apply     # perform the changes

Safety:
- Dry-run by default (unless --apply).
- When applying, creates .bak backups for any edited text files.
- Only touches *.py, *.md, *.tex, *.txt. No binaries.

Notes tailored to your repo:
- Keeps the latest “plot_variance_figures.py” as plot_variance_figures.py.
- Drops older siblings automatically if you don’t copy them in.
"""

import argparse
import re
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]  # ...\JWST-HighZ-Mature-Galaxies\src
FIG_DIR = ROOT / "figures"
ANA_DIR = ROOT / "analysis"

VER_RE = re.compile(r"^(?P<base>.+)_v(?P<num>\d+)(?P<letter>[a-z])?\.py$", re.IGNORECASE)
TEXT_FILE_EXTS = {".py", ".md", ".tex", ".txt"}

def find_versioned(py_dir: Path):
    items = []
    if not py_dir.exists():
        return items
    for p in py_dir.glob("*.py"):
        m = VER_RE.match(p.name)
        if m:
            items.append((p, m.groupdict()))
    return items

def plan_targets(files):
    # Group by base (stem without _vN)
    groups = {}
    for p, g in files:
        base = Path(g["base"]).name
        groups.setdefault(base, []).append((p, g))

    mapping = {}
    for base, members in groups.items():
        # sort chronologically by version number then letter
        members.sort(key=lambda x: (int(x[1]["num"]), x[1]["letter"] or ""))
        if len(members) == 1:
            src = members[0][0]
            dst = src.with_name(f"{base}.py")
            mapping[src] = dst
        else:
            # multiple versions → keep chronological order but map to _a, _b, _c
            for idx, (src, g) in enumerate(members):
                suffix = chr(ord('a') + idx)
                dst = src.with_name(f"{base}_{suffix}.py")
                mapping[src] = dst
    return mapping

def replace_in_text(text, renames):
    # Replace imports/module paths and any filename string literals
    for old_path, new_path in renames.items():
        old_stem = old_path.stem
        new_stem = new_path.stem

        # from figures.old_stem import ...  OR  import figures.old_stem as ...
        pat_mod = re.compile(rf"(\bfrom\s+figures\.|import\s+figures\.){re.escape(old_stem)}\b")
        text = pat_mod.sub(lambda m: m.group(1) + new_stem, text)

        # bare module references: figures.old_stem
        pat_bare = re.compile(rf"\bfigures\.{re.escape(old_stem)}\b")
        text = pat_bare.sub(f"figures.{new_stem}", text)

        # general string replacements for filenames containing old_stem
        text = re.sub(rf"{re.escape(old_stem)}", new_stem, text)

        # also normalize occurrences like base_v5b → base or base_b
        m = VER_RE.match(old_stem + ".py")
        if m:
            base = Path(m.group("base")).name
            letter = m.group("letter") or ""
            text = re.sub(rf"{re.escape(base)}_v\d+{letter}", new_stem, text)

    return text

def update_file_contents(paths_to_edit, renames, dry_run=True):
    edited = []
    for p in paths_to_edit:
        if p.suffix.lower() not in TEXT_FILE_EXTS:
            continue
        try:
            orig = p.read_text(encoding="utf-8")
        except Exception:
            continue
        new = replace_in_text(orig, renames)
        if new != orig:
            if dry_run:
                edited.append(p)
            else:
                bak = p.with_suffix(p.suffix + ".bak")
                if not bak.exists():
                    shutil.copy2(p, bak)
                p.write_text(new, encoding="utf-8")
                edited.append(p)
    return edited

def collect_all_text_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_FILE_EXTS]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Perform changes.")
    ap.add_argument("--dry-run", action="store_true", help="Force dry-run even with --apply.")
    args = ap.parse_args()

    dry_run = not args.apply or args.dry_run

    candidates = []
    candidates += find_versioned(FIG_DIR)
    candidates += find_versioned(ANA_DIR)

    if not candidates:
        print("No versioned scripts found (pattern *_vN[a]?.py). Nothing to do.")
        return 0

    renames = plan_targets(candidates)

    print("Planned renames:")
    for src, dst in sorted(renames.items()):
        try:
            rel_src = src.relative_to(ROOT)
            rel_dst = dst.relative_to(ROOT)
        except Exception:
            rel_src, rel_dst = src, dst
        print(f"  {rel_src}  ->  {rel_dst}")

    # Preview content edits
    all_text = collect_all_text_files(ROOT)
    edited_preview = update_file_contents(all_text, renames, dry_run=True)
    print(f"\nFiles that would be updated for imports/strings: {len(edited_preview)}")
    for p in sorted(edited_preview):
        try:
            rel = p.relative_to(ROOT)
        except Exception:
            rel = p
        print(f"  edit: {rel}")

    if dry_run:
        print("\nDRY-RUN ONLY. To apply, run:\n  python tools\\rename_for_release.py --apply")
        return 0

    # Ensure no collisions
    collisions = [dst for dst in renames.values() if dst.exists()]
    if collisions:
        print("\nERROR: The following target files already exist; resolve before applying:")
        for c in collisions:
            try:
                print(f"  {c.relative_to(ROOT)}")
            except Exception:
                print(f"  {c}")
        return 2

    # Perform renames
    for src, dst in renames.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        try:
            print(f"RENAMED: {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        except Exception:
            print(f"RENAMED: {src} -> {dst}")

    # Apply content edits repo-wide
    edited = update_file_contents(all_text, renames, dry_run=False)
    print(f"\nEdited {len(edited)} files (created .bak backups).")
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
