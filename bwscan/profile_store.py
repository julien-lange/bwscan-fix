"""Persist pick history and learn default clip values (JSON).

Picks from every reviewed roll live in one global JSON file
(``~/.config/bwscan/profile.json`` by default; override with ``--profile-dir``).
Each pick is tagged with the roll it came from so same-named scans across
rolls don't collide. Once enough non-custom picks exist, a learned default
(median black/white clip, modal curve over the recent picks) is computed —
that default is what powers ``batch``/``--default`` on future rolls.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

MIN_PICKS_DEFAULT = 5
MAX_PICKS_KEPT = 400

LEARN_WINDOW = 12


def default_profile_dir() -> Path:
    return Path(os.environ.get("BWSCAN_PROFILE_DIR", "~/.config/bwscan")).expanduser()


class ProfileStore:
    def __init__(
        self,
        roll: str | Path,
        profile_dir: Optional[str | Path] = None,
        min_picks: int = MIN_PICKS_DEFAULT,
    ):
        self.roll = Path(roll).name
        self.min_picks = min_picks
        self.profile_dir = Path(profile_dir) if profile_dir else default_profile_dir()
        self.path = self.profile_dir / "profile.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"picks": [], "learned": None}

    def save(self) -> None:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def picks(self) -> List[dict]:
        return self._data["picks"]

    def record_pick(
        self,
        filename: str,
        black_pct: float,
        white_pct: float,
        curve: str,
        tile: int,
        variant_label: str,
        custom: bool = False,
        gamma: float = 1.0,
        s_weight: float = 0.0,
    ) -> None:
        self.picks.append(
            {
                "roll": self.roll,
                "file": filename,
                "black_pct": round(float(black_pct), 4),
                "white_pct": round(float(white_pct), 4),
                "curve": curve,
                "gamma": round(float(gamma), 4),
                "s_weight": round(float(s_weight), 4),
                "tile": tile,
                "variant_label": variant_label,
                "custom": bool(custom),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        if len(self.picks) > MAX_PICKS_KEPT:
            self._data["picks"] = self.picks[-MAX_PICKS_KEPT:]
        self.maybe_learn()
        self.save()

    def pick_for(self, roll: str, filename: str) -> Optional[dict]:
        for pick in reversed(self.picks):
            if pick.get("roll") == roll and pick["file"] == filename:
                return pick
        return None

    def maybe_learn(self) -> bool:
        """Recompute the learned default from recent non-custom picks."""
        recent = [
            p for p in self.picks if not p.get("custom")
        ][-LEARN_WINDOW:]
        if len(recent) < self.min_picks:
            return False
        blacks = sorted(p["black_pct"] for p in recent)
        whites = sorted(p["white_pct"] for p in recent)
        mid = len(recent) // 2
        curve = Counter(p["curve"] for p in recent).most_common(1)[0][0]
        gamma = Counter(round(p.get("gamma", 1.0), 3) for p in recent).most_common(1)[0][0]
        s_weight = Counter(
            round(p.get("s_weight", 0.0), 3) for p in recent
        ).most_common(1)[0][0]
        self._data["learned"] = {
            "black_pct": round(float(blacks[mid]), 4),
            "white_pct": round(float(whites[mid]), 4),
            "curve": curve,
            "gamma": round(float(gamma), 4),
            "s_weight": round(float(s_weight), 4),
            "n": len(recent),
            "min_picks": self.min_picks,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return True

    @property
    def learned(self) -> Optional[dict]:
        return self._data.get("learned")

    def describe(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "roll": self.roll,
            "total_picks": len(self.picks),
            "custom_picks": sum(1 for p in self.picks if p.get("custom")),
            "min_picks": self.min_picks,
            "learned": self.learned,
        }