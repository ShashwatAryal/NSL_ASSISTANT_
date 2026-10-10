"""I/O and naming utilities for NSL landmark clips and metadata records.

This module provides functions to validate, save, load, and parse .npz
landmark clips according to the contract, and helpers to manage metadata.csv.
"""

import csv
from pathlib import Path
import re
import numpy as np

from nsl.config import METADATA_PATH, SIGNS

# Metadata table header columns in the exact order specified by the contract
METADATA_COLUMNS = [
    "filename",
    "sign",
    "person",
    "source",
    "session",
    "lighting",
    "notes",
    "n_frames",
    "fps",
]

# Regular expressions for clip filenames
# 1. Reference: ref_<sign>_<source>_<nn>  e.g. ref_fever_sanketik_01
_REF_PATTERN = re.compile(
    r"^ref_(?P<sign>[a-z]+)_(?P<source>[a-zA-Z0-9]+)_(?P<index>\d+)$"
)

# 2. Own clips: <person>_<sign>_s<session>_<nnn>  e.g. p2_fever_s1_007
_OWN_PATTERN = re.compile(
    r"^(?P<person>p\d+)_(?P<sign>[a-z]+)_s(?P<session>[a-zA-Z0-9]+)_(?P<index>\d+)$"
)


def validate_clip(clip: dict) -> None:
    """Validate clip structure, shapes, and data types against the contract.

    Args:
        clip: Dictionary containing landmarks, detections, fps, and source.

    Raises:
        ValueError: If any key is missing or has an incorrect shape or dtype.
    """
    required_keys = ["hands", "pose", "hand_present", "pose_present", "fps", "source"]
    for key in required_keys:
        if key not in clip:
            raise ValueError(f"Missing required key '{key}' in clip dictionary.")

    # 1. Validate hands: shape (T, 2, 21, 3), float32
    hands = clip["hands"]
    if not isinstance(hands, np.ndarray):
        raise ValueError("Key 'hands' must be a numpy ndarray.")
    if hands.dtype != np.float32:
        raise ValueError(f"Key 'hands' must have dtype float32, got {hands.dtype}.")
    if hands.ndim != 4 or hands.shape[1:] != (2, 21, 3):
        raise ValueError(
            f"Key 'hands' must have shape (T, 2, 21, 3), got {hands.shape}."
        )

    t_frames = hands.shape[0]

    # 2. Validate pose: shape (T, 9, 3), float32
    pose = clip["pose"]
    if not isinstance(pose, np.ndarray):
        raise ValueError("Key 'pose' must be a numpy ndarray.")
    if pose.dtype != np.float32:
        raise ValueError(f"Key 'pose' must have dtype float32, got {pose.dtype}.")
    if pose.shape != (t_frames, 9, 3):
        raise ValueError(
            f"Key 'pose' must have shape ({t_frames}, 9, 3), got {pose.shape}."
        )

    # 3. Validate hand_present: shape (T, 2), bool
    hand_present = clip["hand_present"]
    if not isinstance(hand_present, np.ndarray):
        raise ValueError("Key 'hand_present' must be a numpy ndarray.")
    if hand_present.dtype != bool and hand_present.dtype != np.bool_:
        raise ValueError(
            f"Key 'hand_present' must have dtype bool, got {hand_present.dtype}."
        )
    if hand_present.shape != (t_frames, 2):
        raise ValueError(
            f"Key 'hand_present' must have shape ({t_frames}, 2), got {hand_present.shape}."
        )

    # 4. Validate pose_present: shape (T,), bool
    pose_present = clip["pose_present"]
    if not isinstance(pose_present, np.ndarray):
        raise ValueError("Key 'pose_present' must be a numpy ndarray.")
    if pose_present.dtype != bool and pose_present.dtype != np.bool_:
        raise ValueError(
            f"Key 'pose_present' must have dtype bool, got {pose_present.dtype}."
        )
    if pose_present.shape != (t_frames,):
        raise ValueError(
            f"Key 'pose_present' must have shape ({t_frames},), got {pose_present.shape}."
        )

    # 5. Validate fps: positive number
    fps = clip["fps"]
    if not isinstance(fps, (int, float, np.floating, np.integer)) or fps <= 0:
        raise ValueError(f"Key 'fps' must be a positive number, got {fps}.")

    # 6. Validate source: non-empty string
    source = clip["source"]
    if not isinstance(source, (str, np.str_)) or len(str(source)) == 0:
        raise ValueError(f"Key 'source' must be a non-empty string, got {source}.")


def save_clip(path: str | Path, clip: dict) -> None:
    """Validate and save a landmark clip to a compressed .npz file.

    Args:
        path: File destination path for the saved clip (.npz).
        clip: Dictionary adhering to the clip format specification.

    Raises:
        ValueError: If clip data does not match contract shapes or dtypes..
    """
    validate_clip(clip)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        file_path,
        hands=clip["hands"],
        pose=clip["pose"],
        hand_present=clip["hand_present"],
        pose_present=clip["pose_present"],
        fps=float(clip["fps"]),
        source=str(clip["source"]),
    )


def load_clip(path: str | Path) -> dict:
    """Load, validate, and return a landmark clip from an .npz file.

    Args:
        path: Path to the .npz clip file to load.

    Returns:
        Dictionary with validated arrays, float fps, and string source.

    Raises:
        FileNotFoundError: If the specified clip file does not exist.
        ValueError: If the file contents violate the contract format.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Clip file not found: {file_path}")

    with np.load(file_path) as data:
        source_val = data["source"]
        # Handle scalar string extracted from numpy array
        source_str = str(source_val.item() if hasattr(source_val, "item") else source_val)

        clip = {
            "hands": np.asarray(data["hands"], dtype=np.float32),
            "pose": np.asarray(data["pose"], dtype=np.float32),
            "hand_present": np.asarray(data["hand_present"], dtype=bool),
            "pose_present": np.asarray(data["pose_present"], dtype=bool),
            "fps": float(data["fps"]),
            "source": source_str,
        }

    validate_clip(clip)
    return clip


def parse_filename(name: str) -> dict:
    """Parse a clip filename (stem or path) and return its metadata components.

    Args:
        name: Filename string, with or without path or file extension.

    Returns:
        Dict with keys: kind ("ref" or "own"), sign, person, source, session, index.

    Raises:
        ValueError: If name matches neither pattern or the sign is not in SIGNS.
    """
    stem = Path(name).stem

    # Check reference pattern: ref_<sign>_<source>_<nn>
    ref_match = _REF_PATTERN.match(stem)
    if ref_match:
        sign = ref_match.group("sign")
        if sign not in SIGNS:
            raise ValueError(f"Unknown sign '{sign}' in '{name}'. Must be in SIGNS.")
        return {
            "kind": "ref",
            "sign": sign,
            "person": "ref",
            "source": ref_match.group("source"),
            "session": "",
            "index": int(ref_match.group("index")),
        }

    # Check own pattern: <person>_<sign>_s<session>_<nnn>
    own_match = _OWN_PATTERN.match(stem)
    if own_match:
        sign = own_match.group("sign")
        if sign not in SIGNS:
            raise ValueError(f"Unknown sign '{sign}' in '{name}'. Must be in SIGNS.")
        return {
            "kind": "own",
            "sign": sign,
            "person": own_match.group("person"),
            "source": "own",
            "session": own_match.group("session"),
            "index": int(own_match.group("index")),
        }

    raise ValueError(
        f"Filename '{name}' does not match reference or own naming pattern."
    )


def build_own_filename(person: str, sign: str, session: int | str, index: int) -> str:
    """Build the standard filename stem for an own recorded clip.

    Example:
        build_own_filename("p2", "fever", "1", 7) -> "p2_fever_s1_007"

    Args:
        person: Identifier string for signer (e.g., 'p1', 'p2').
        sign: Name of the sign (must belong to SIGNS).
        session: Session identifier or number (e.g., 1 or 's1').
        index: Sequential recording index (formatted as 3 digits).

    Returns:
        Filename stem string without extension.

    Raises:
        ValueError: If the sign is not in SIGNS.
    """
    if sign not in SIGNS:
        raise ValueError(f"Sign '{sign}' is not recognized in SIGNS: {SIGNS}.")

    sess_str = str(session)
    if not sess_str.startswith("s"):
        sess_str = f"s{sess_str}"

    return f"{person}_{sign}_{sess_str}_{int(index):03d}"


def next_index(
    directory: str | Path, person: str, sign: str, session: int | str
) -> int:
    """Find the next free recording index counter for a given signer, sign, and session.

    Args:
        directory: Directory containing saved .npz landmark files.
        person: Signer identifier string (e.g., 'p1').
        sign: Sign name string.
        session: Session identifier (e.g., 1 or 's1').

    Returns:
        The next sequential integer counter, starting at 1 if none exist.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return 1

    matching_indices: list[int] = []
    sess_norm = str(session).lstrip("s")

    for file in dir_path.glob("*.npz"):
        try:
            parsed = parse_filename(file.name)
        except ValueError:
            continue

        if (
            parsed["kind"] == "own"
            and parsed["person"] == person
            and parsed["sign"] == sign
            and str(parsed["session"]).lstrip("s") == sess_norm
        ):
            matching_indices.append(parsed["index"])

    if not matching_indices:
        return 1
    return max(matching_indices) + 1


def read_metadata(path: str | Path | None = None) -> list[dict]:
    """Read metadata CSV rows into a list of row dictionaries.

    Args:
        path: Path to metadata CSV file. If None, uses METADATA_PATH from config.

    Returns:
        List of dicts representing metadata rows, or an empty list if file is absent.
    """
    file_path = Path(path) if path is not None else METADATA_PATH
    if not file_path.is_file():
        return []

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_metadata(row: dict, path: str | Path | None = None) -> None:
    """Append a single record row to the metadata CSV file.

    Creates the file and writes the header if the file does not already exist.

    Args:
        row: Dictionary containing metadata columns.
        path: Path to metadata CSV file. If None, uses METADATA_PATH from config.
    """
    file_path = Path(path) if path is not None else METADATA_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not file_path.is_file() or file_path.stat().st_size == 0

    with file_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def remove_metadata_row(filename: str, path: str | Path | None = None) -> bool:
    """Remove a row with the given filename stem from the metadata CSV file.

    Args:
        filename: Stem (without extension) or file name to match.
        path: Path to metadata CSV file. If None, uses METADATA_PATH from config.

    Returns:
        True if at least one matching row was found and removed, False otherwise.
    """
    file_path = Path(path) if path is not None else METADATA_PATH
    if not file_path.is_file():
        return False

    stem = Path(filename).stem
    rows = read_metadata(file_path)
    kept_rows = [r for r in rows if r.get("filename") != stem]

    if len(kept_rows) == len(rows):
        return False

    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept_rows)
    return True

