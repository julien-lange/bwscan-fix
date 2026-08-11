# bwscan

Terminal-based optimizer for B&W negative scans, tuned for my fixed
pipeline: **Kentmere 400** in a **Pentax Spotmatic ES2**, scanned on a
**Plustek 7200** via **VueScan** (16-bit RGB TIFF, whole roll batches).

The idea: since the camera/film/scanner never changes, the tool doesn't need
to be a general raw processor. It proposes a small, consistent grid of tonal
remaps per photo (black/white percentile clips, plus one mild S-curve), you
pick the best one in the terminal while it opens a contact-sheet grid in
Preview, and it learns which clip values suit the setup so future rolls can
be handled with a single command and a quick outlier check.

## Install

```sh
cd bwscan
python3 -m venv .venv
.venv/bin/pip install -e .
# then either:
.venv/bin/bwscan --help        # via the venv
# or symlink .venv/bin/bwscan onto your PATH, or `pipx install .`
```

## Workflow

### 1. Sanity-check a roll

```
bwscan analyze scans/roll01
```

Prints min/percentiles/max/mean/stddev per scan plus a compact text
histogram, and caches the analysis to `scans/roll01/.bwscan_cache/` so
re-runs are instant. Use this first to confirm VueScan output is flat and
the histograms look sane.

### 2. Interactive review (the first few rolls)

```
bwscan review scans/roll01
```

For each photo it renders a 6-tile grid (five percentile-clip variants with
progressively stronger contrast, plus one full S-curve) to
`review/<file>_grid.png`, opens it in Preview, and prompts:

```
  Pick [1-6] or c/s/q:
```

- `1-6`  pick that variant; recorded in the profile
- `c`     type your own black%/white% clip, re-rendered as tile 7 and reopened
- `s`     skip (scan passes through unprocessed)
- `q`     quit (remaining photos are left for a later run)

The grid window is automatically closed in Preview once you make a choice, so
Preview doesn't pile up one window per photo across the roll.

Pass `-a/--apply` to write the full-resolution outputs to `output/` as soon
as the review ends.

After enough picks (default `-n 5`), the profile computes a **learned
default** — median black/white clip, modal curve.

### 3. Fast mode (once you trust the default)

```
bwscan batch scans/rollN -o output
```

Applies the learned default to the whole new roll, writes a
`output/verify_batch.png` contact sheet, and opens it so you can spot
outliers (very contrasty or very thin negatives). Re-run any stragglers
through `bwscan review`.

### 4. Apply / inspect

```
bwscan apply  scans/roll01            # writes output/ from recorded picks
bwscan apply  scans/roll02 --default  # learned default where no pick exists
bwscan profile
```

`apply` re-derives the exact clip pixel values from the loaded 16-bit array,
so results match the previews to the pixel. Scans with no pick and no usable
default are copied through byte-for-byte. `--jpeg` also exports 8-bit JPEGs
for quick sharing.

## Configuration

- Picks and the learned default live in `~/.config/bwscan/profile.json`
  (override with `--profile-dir PATH`, or the `BWSCAN_PROFILE_DIR` env var).
  Picks are tagged with their roll and filename.
- Grid variants are defined in `bwscan/variants.py` (`DEFAULT_GRID`). Each is
  a named look combining a black/white clip, a `gamma` midtone lift/darken
  and an `s_weight` S-curve strength — tweak these once you've seen a few
  real rolls.
- Cache dir `<folder>/.bwscan_cache/` holds per-image stats; it's keyed on
  file size+mtime so editing or re-scanning invalidates it automatically.

## Development

```sh
.venv/bin/python -m pytest
```

Run `analyze` on a real roll first — if the 0.05–1% / 99–99.95% clip range
looks off for your scans, adjust `DEFAULT_GRID` before running review loops.

## Layout

```
bwscan/            CLI entry point
bwscan/cli.py      argument parsing + command dispatch
bwscan/io_tiff.py  16-bit TIFF load/save (RGB scans -> grayscale array)
bwscan/stats.py    histogram + percentile analysis, JSON cache
bwscan/variants.py candidate remaps + LUT application
bwscan/grid.py     labeled contact-sheet PNGs
bwscan/review.py   terminal picking loop (selection, c = customize)
bwscan/profile_store.py  pick history + learned default (JSON)
bwscan/apply.py    full-resolution output
tests/             pytest suite (synthetic scans)
```
