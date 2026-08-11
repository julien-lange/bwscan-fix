"""Command-line entry point.

Commands
--------
analyze   FOLDER   print per-scan histogram/percentile stats (sanitises cache)
review    FOLDER   interactive grid review; records picks in the profile
batch     FOLDER   fast mode: apply the learned default + verify contact sheet
apply     FOLDER   write full-resolution outputs from picks (or learned default)
profile   FOLDER   show pick history and the learned default
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import apply as apply_mod
from . import grid as grid_mod
from . import review as review_mod
from . import stats
from . import profile_store

DEFAULT_FOLDER = "."


def _analyze(folder: Path, args) -> int:
    tiffs = apply_mod.find_tiffs(folder)
    if not tiffs:
        print(f"No TIFFs found in {folder}")
        return 1

    rows: List[dict] = []
    for tiff in tiffs:
        stati = stats.analyze_tiff(tiff, use_cache=not args.no_cache)
        rows.append({"name": tiff.name, "stats": stati})

    print(f"{'file':<24}{'w x h':<14}{'min':>6}{'p0.1':>8}{'p1':>7}{'p50':>7}"
          f"{'p99':>7}{'p99.9':>8}{'max':>7}{'mean':>8}{'sd':>7}")
    for r in rows:
        s = r["stats"]
        print(f"{r['name']:<24}{s.width}x{s.height:<9}{s.min:>6}"
              f"{s.percentiles['0.1']:>8}{s.percentiles['1.0']:>7}"
              f"{s.percentiles['50.0']:>7}{s.percentiles['99.0']:>7}"
              f"{s.percentiles['99.9']:>8}{s.max:>7}{s.mean:>8.0f}{s.stddev:>7.0f}")
    print("\nCompact histograms (log scale, values at each bucket):")
    for r in rows:
        print(f"\n{r['name']}:")
        print("\n".join(stats.ascii_histogram(r["stats"])))
    return 0


def _review(folder: Path, args) -> int:
    return review_mod.review_batch(
        folder,
        no_open=args.no_open,
        apply_after=args.apply,
        output_dir=args.output,
        min_picks=args.min_picks,
        profile_dir=args.profile_dir,
    )


def _batch(folder: Path, args) -> int:
    profile = profile_store.ProfileStore(
        roll=folder, profile_dir=args.profile_dir, min_picks=args.min_picks
    )
    if profile.learned is None:
        print(
            f"No learned default yet (need >= {args.min_picks} picks).\n"
            "Run `bwscan review` until the profile learns one, then retry."
        )
        return 1
    out = Path(args.output)
    t0 = time.time()
    results = apply_mod.apply_batch(
        folder, out, use_default=True, jpeg=args.jpeg, profile=profile
    )
    applied = sum(1 for _, s in results if s == "applied")
    unchanged = sum(1 for _, s in results if s == "unchanged")
    print(f"applied {applied}, unchanged {unchanged} "
          f"({apply_mod.elapsed(t0)}), using learned default from {len(profile.picks)} picks")
    _verify_sheet(out, args.no_open)
    return 0


def _verify_sheet(out: Path, no_open: bool) -> Optional[Path]:
    tiles: List[tuple] = []
    for tiff in sorted(out.glob("*.tif")) + sorted(out.glob("*.tiff")):
        from . import io_tiff
        try:
            data = io_tiff.read_tiff(tiff)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {tiff.name}: {exc}")
            continue
        thumb = grid_mod.to_pil_rgb(data, 380)
        tiles.append((f"{tiff.name}", thumb))
    if not tiles:
        print("  nothing to show")
        return None
    sheet = grid_mod.make_grid(tiles, cols=4, header="Batch verify")
    path = grid_mod.write_grid(sheet, out, "verify_batch.png", parent=".")
    if not no_open:
        grid_mod.open_image(path)
    else:
        print(f"[verify] {path}")
    return path


def _apply(folder: Path, args) -> int:
    profile = profile_store.ProfileStore(
        roll=folder, profile_dir=args.profile_dir, min_picks=args.min_picks
    )
    t0 = time.time()
    results = apply_mod.apply_batch(
        folder,
        args.output,
        use_default=args.default,
        jpeg=args.jpeg,
        profile=profile,
    )
    applied = sum(1 for _, s in results if s == "applied")
    unchanged = sum(1 for _, s in results if s == "unchanged")
    print(f"{len(results)} files -> {args.output}/ "
          f"(applied {applied}, unchanged {unchanged}, {apply_mod.elapsed(t0)})")
    for tiff, status in results:
        print(f"    {status:9s} {tiff.name}")
    return 0


def _profile(folder: Path, args) -> int:
    profile = profile_store.ProfileStore(
        roll=folder, profile_dir=args.profile_dir, min_picks=args.min_picks
    )
    d = profile.describe()
    print(f"profile:       {d['path']}")
    print(f"roll:          {d['roll']}")
    print(f"total picks:   {d['total_picks']}")
    print(f"custom picks:  {d['custom_picks']}")
    print(f"min picks:     {d['min_picks']}")
    print("learned:")
    learned = d["learned"]
    if learned:
        print(f"  black={learned['black_pct']}%  white={learned['white_pct']}%"
              f"  gamma={learned.get('gamma', 1.0)}  curve={learned['curve']}"
              f"  (n={learned['n']}, updated {learned['updated']})")
    else:
        print("  none yet")
    print("last 10 picks:")
    for p in profile.picks[-10:]:
        mark = "custom" if p["custom"] else p["variant_label"]
        print(f"  {p['time']}  {p['file']:<28} tile {p['tile']:>2} "
              f"(b={p['black_pct']}, w={p['white_pct']}, g={p.get('gamma', 1.0)}, "
              f"{p['curve']}, {mark})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bwscan",
        description="B&W negative scan optimizer (Kentmere 400 / Plustek 7200).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, extra=()):
        sp.add_argument("folder", nargs="?", default=DEFAULT_FOLDER,
                        help="folder with TIFF scans (default: current dir)")
        sp.add_argument("-o", "--output", default="output",
                        help="output folder (default: output/)")
        sp.add_argument("--no-open", action="store_true",
                        help="do not auto-open grids in Preview")
        sp.add_argument("--jpeg", action="store_true",
                        help="also export 8-bit JPEG per applied scan")
        sp.add_argument("--profile-dir", default=None,
                        help="where profile.json lives (default: ~/.config/bwscan)")
        for name, kwargs in extra:
            sp.add_argument(name, **kwargs)

    sp = sub.add_parser("analyze", help="print per-scan stats")
    add_common(sp)
    sp.add_argument("--no-cache", action="store_true", help="recompute stats")

    sp = sub.add_parser("review", help="interactive grid review loop")
    add_common(sp)
    sp.add_argument("-a", "--apply", action="store_true",
                    help="apply selections to output/ at the end")
    sp.add_argument("-n", "--min-picks", type=int,
                    default=profile_store.MIN_PICKS_DEFAULT,
                    help="picks before a default is learned")

    sp = sub.add_parser("batch", help="fast mode (learned default + verify sheet)")
    add_common(sp)
    sp.add_argument("-n", "--min-picks", type=int,
                    default=profile_store.MIN_PICKS_DEFAULT,
                    help="picks before a default is learned")

    sp = sub.add_parser("apply", help="write full-resolution outputs")
    add_common(sp)
    sp.add_argument("-d", "--default", action="store_true",
                    help="apply learned default where no pick exists")
    sp.add_argument("-n", "--min-picks", type=int,
                    default=profile_store.MIN_PICKS_DEFAULT,
                    help="picks before a default is learned")

    sp = sub.add_parser("profile", help="show pick history + learned default")
    add_common(sp)
    sp.add_argument("-n", "--min-picks", type=int,
                    default=profile_store.MIN_PICKS_DEFAULT)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        parser.error(f"not a directory: {folder}")
    if args.command == "analyze":
        return _analyze(folder, args)
    if args.command == "review":
        return _review(folder, args)
    if args.command == "batch":
        return _batch(folder, args)
    if args.command == "apply":
        return _apply(folder, args)
    if args.command == "profile":
        return _profile(folder, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())