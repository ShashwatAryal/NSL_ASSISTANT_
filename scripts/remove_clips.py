"""Safely remove recorded clips (moves them to a backup folder, never hard-deletes).

Examples (run from the project root):
    python -m scripts.remove_clips p1_no_s1_005 p1_no_s1_006
    python -m scripts.remove_clips --signs no yes vomiting --person p1
    python -m scripts.remove_clips --signs no --person p1 --dry-run

What it does for every selected clip:
  1. moves data/landmarks/own/<clip>.npz into data/removed/<timestamp>/
  2. moves the raw video too, if one was saved
  3. removes the clip's row from data/metadata.csv (a backup copy of the
     metadata is saved first)

Only YOUR OWN recorded clips can be removed. Reference clips (ref_...) are
refused on purpose. To undo, move the files back from data/removed/ and
restore the metadata backup.
"""

import argparse
import shutil
import time
from pathlib import Path

from nsl.clipio import read_metadata, remove_metadata_row
from nsl.config import LANDMARKS_OWN_DIR, METADATA_PATH, RAW_OWN_DIR


def select_stems(args, rows: list[dict]) -> tuple[list[str], list[str]]:
    """Work out which clip names to remove.

    Returns (stems, refused) where refused are names that were not allowed.
    """
    named = {Path(n).stem for n in args.clips}
    refused = sorted(n for n in named if n.startswith("ref_"))
    named -= set(refused)

    stems = set(named)
    if args.signs:
        for row in rows:
            if row.get("source") != "own":
                continue
            if row.get("sign") not in args.signs:
                continue
            if args.person and row.get("person") not in args.person:
                continue
            if args.session and row.get("session") not in args.session:
                continue
            stems.add(row["filename"])
    return sorted(stems), refused


def describe(stems: list[str], rows: list[dict], landmarks_dir: Path) -> None:
    """Print what will be removed and whether each file and row exists."""
    in_meta = {r["filename"] for r in rows}
    print(f"\n{'clip':<28}{'file':>8}{'metadata':>10}")
    for stem in stems:
        file_ok = (landmarks_dir / f"{stem}.npz").is_file()
        print(f"{stem:<28}{'yes' if file_ok else 'MISSING':>8}"
              f"{'yes' if stem in in_meta else 'none':>10}")


def move_clip(stem: str, landmarks_dir: Path, raw_dir: Path, backup_dir: Path) -> None:
    """Move one clip's files into the backup folder."""
    for src in (landmarks_dir / f"{stem}.npz", raw_dir / f"{stem}.mp4"):
        if src.is_file():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(backup_dir / src.name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Move unwanted own clips to a backup folder.")
    parser.add_argument("clips", nargs="*", help="Clip names, e.g. p1_no_s1_005")
    parser.add_argument("--signs", nargs="+", help="Remove clips of these signs.")
    parser.add_argument("--person", nargs="+", help="Limit --signs to these people.")
    parser.add_argument("--session", nargs="+", help="Limit --signs to these sessions, e.g. 1.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen only.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation question.")
    parser.add_argument("--landmarks-dir", type=Path, default=LANDMARKS_OWN_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_OWN_DIR)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    args = parser.parse_args()

    if not args.clips and not args.signs:
        parser.print_help()
        return

    rows = read_metadata(args.metadata)
    stems, refused = select_stems(args, rows)
    for name in refused:
        print(f"Refused: {name} (reference clips cannot be removed with this tool)")
    if not stems:
        print("No clips selected. Nothing to do.")
        return

    describe(stems, rows, args.landmarks_dir)
    if args.dry_run:
        print(f"\nDry run: {len(stems)} clip(s) would be moved. Nothing was changed.")
        return

    if not args.yes:
        answer = input(f"\nMove {len(stems)} clip(s) to a backup folder? Type yes to confirm: ")
        if answer.strip().lower() != "yes":
            print("Cancelled. Nothing was changed.")
            return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = args.metadata.parent / "removed" / stamp
    if args.metadata.is_file():                       # keep a copy of the metadata first
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.metadata, backup_dir / "metadata_before.csv")

    removed_rows = 0
    for stem in stems:
        move_clip(stem, args.landmarks_dir, args.raw_dir, backup_dir)
        if remove_metadata_row(stem, args.metadata):
            removed_rows += 1

    print(f"\nMoved {len(stems)} clip(s) to {backup_dir}")
    print(f"Removed {removed_rows} metadata row(s).")


if __name__ == "__main__":
    main()
