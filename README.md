# HCNS_data_reduction

Code for reducing HST imaging for the Hubble Census of Nearby Satellites
(HCNS) survey, plus archival HST data for HCNS-relevant dwarf galaxies from
other programmes. The pipeline runs in four stages, each a standalone script
run from this directory:

1. **`HCNS_download.py`** — download raw HST data from MAST.
2. **`HCNS_dolphot.py`** — run dolphot photometry (prep, alignment, AST).
3. **`HCNS_first_CMDs.py`** — quality-cut photometry, extinction-correct,
   produce colour–magnitude diagrams and completeness curves.
4. **`HCNS_RGB_images.py`** — produce greyscale and RGB composite images.

All four operate on the same set of targets and expect to be run in this
order from within `HCNS_data_reduction/`, with sibling `data/`, `reduction/`
and `output/` directories one level up.

## Directory layout

```
../data/<target>/                         raw FITS (HCNS survey)
../data/archival/<proposal_id>/<target>/  raw FITS (archival)
../reduction/<target>/                    dolphot working directory (HCNS)
../reduction/archival/<proposal_id>/<target>/   dolphot working dir (archival)
../output/<target>/                       catalogs, CMDs, RGB images (HCNS)
../output/archival/<proposal_id>/<target>/      same, for archival
```

Every script accepts this same two-mode split: **HCNS mode** (the main
survey, default) and **archival mode** (`--archival`), which processes
`../data/archival/<proposal_id>/<target>/` instead.

---

## 1. `HCNS_download.py`

Downloads DRC (drizzled mosaic, used as the dolphot reference image), FLC
(CTE-corrected exposures) and FLT (non-CTE exposures) products from MAST.

**Flags**
- `--targets` — HCNS mode only. Restrict to observations whose `dataURL`
  matches an entry in `good_obs.list`.
- `--archival` — switch to archival mode (see below).

**HCNS mode (default):** queries proposal 18061 ("Hubble Census of Nearby
Satellites"), excludes anything in `bad_obs.list`, downloads one directory
per target under `../data/`.

**Archival mode (`--archival`)** runs two independent phases, in order:

1. **TSV-driven targeted downloads** (runs first) — if
   `HCNS_Archival_Dwarfs.tsv` is present (not tracked in git — a per-machine
   reference table), reads each row with a non-blank `Codes (Observation ID)`
   column. Each usable row already specifies exact HST rootname codes and a
   proposal ID, so this path downloads directly by matching those codes
   against `dataURL`, without needing the HCNS-sample cross-match. MAST is
   queried once per unique proposal ID (cached across rows). Any
   `Other_Names`/`HCNS_Name` mismatch is appended to `archival_name_map.list`
   automatically.
2. **Whole-project scan** (runs second) — for each proposal ID in
   `archival_projects.list`, queries the entire proposal, excludes
   `archival_bad_obs.list`, maps MAST target names to canonical HCNS names
   via `archival_name_map.list` (case-insensitive), cross-matches against the
   HCNS sample Google Sheet, and optionally restricts to
   `archival_targets.list` if non-empty.

**Exclusion is applied twice** for robustness: once at the MAST observation-
row level (cheap, filters most rows before any product-list query), and
again at the individual product-filename level right before download. The
second check is necessary because MAST can expose the same exposure through
more than one observation record (e.g. a classic per-exposure record versus
a newer HAP-association record for the same visit), only one of which has a
`dataURL` that literally names the exposure — the row-level check alone can
miss files that come through via the other record.

Ctrl+C requests a graceful stop after the current target finishes; a second
Ctrl+C forces an immediate exit.

---

## 2. `HCNS_dolphot.py`

Runs the dolphot photometry pipeline per target: copy raw files, mask,
`splitgroups`, `calcsky`, align, run photometry, generate artificial star
tests (ASTs).

**Flags**
- `--archival` — process `../data/archival/` instead of `../data/`.
- `--ast` — run 9 additional AST iterations per target (suffixed `_01`
  through `_09`), each producing its own `.fake` catalog. Requires the
  standard AST run to have completed first; targets without `dolphot.done`
  are silently skipped (run without `--ast` first to process new targets).
- `--ncpu` — number of simultaneous dolphot processes (default 10). Lower
  this if running `--ast` locks up the machine.

**Archival parallelism.** In `--archival` mode, targets from every archival
project are flattened into a single pool before each stage (prep, dolphot,
AST) is parallelised, so `--ncpu` workers are shared across *all* archival
programs at once rather than being confined to one program's targets at a
time. This matters because most individual archival programs only have 1-2
targets — without flattening, most of `--ncpu` would sit idle while a
single-target program ran alone before moving to the next.

**Filter selection.** dolphot's `fakelist` tool only supports two-band AST
generation, so before any files are copied from the download directory,
`prep_dolphot` selects exactly two filters: `F814W` is always the red
anchor; the blue band is the first filter found, in order, from
`BLUE_FILTER_PREFERENCE = ['F606W', 'F555W', 'F475W']` (defined near the top
of the file — edit this list to add or reorder acceptable blue filters).
Targets without both `F814W` and an acceptable blue filter are skipped. This
means dolphot, and everything downstream, only ever sees two filters even
when a target has 3+ bands of raw data.

**Alignment** retries up to 4 times with an escalating `Align` parameter
(2 → 3 → 4 → 2-with-reference-switched-to-shortest-wavelength-filter),
validated by requiring every per-image sigma ≤ 0.75 and every matched-star
count ≥ 15.

**Marker files** (per target, in the reduction directory) gate each step so
re-runs skip completed work: `<instr>mask.done`, `splitgroups.done`,
`calcsky.done`, `align.done` / `align.failed`, `dolphot.done`,
`fakestars.done`, `fakestars_NN.done` (extra `--ast` iterations). Delete the
relevant marker (and any it depends on) to force a re-run of that step.

---

## 3. `HCNS_first_CMDs.py`

For each target with completed dolphot photometry:
- Reads the dolphot catalog and applies quality cuts (source type,
  magnitude, crowding, sharpness).
- Queries the SFD dust map for per-star extinction corrections.
- Saves full-field and target-region (within 2 r_e of the HCNS sample
  position) photometry catalogs and CMD plots.
- If AST results are available, fits completeness-limit curves as a function
  of F814W magnitude and F606W−F814W colour.

**Flag:** `--overwrite` — reprocess outputs even if they already exist.

**AST auto-update.** If `--ast` has produced additional `_01`–`_09`
iterations since the last run, the completeness fit is automatically
refreshed the next time this script runs — no `--overwrite` needed — by
comparing the modification times of the new `.fake` files against the
existing `phot_ast_full.csv`. `--overwrite` only forces a full reprocess of
everything, including the base (non-AST) photometry and CMDs.

Column indices for the dolphot catalog are hard-coded assuming exactly two
filters; this stays valid as long as `HCNS_dolphot.py`'s filter selection
(above) keeps every reduction directory down to two bands.

---

## 4. `HCNS_RGB_images.py`

Produces one greyscale PNG per filter and a three-colour composite PNG per
target, reading DRC images directly from the download directory (not the
dolphot reduction directory).

**Flag:** `--overwrite` — regenerate images even if they already exist.

Supports targets with **2 or 3 filters** (more or fewer are skipped with a
warning). Filters are ordered longest-to-shortest wavelength; the reddest
filter is the alignment reference, and every other filter is aligned onto it
via a three-tier fallback (astroalign triangle-matching → star-cloud ICP →
WCS-only reprojection). For 2-filter targets the green channel is
synthesised as the average of red and blue (as before); for 3-filter targets
the middle-wavelength filter is used as a real green channel.

---

## Configuration / list files

| File | Used by | Purpose |
|---|---|---|
| `bad_obs.list` | download (HCNS mode) | Observation-ID substrings to exclude from the main survey download. |
| `good_obs.list` | download (`--targets`) | Observation-ID substrings to restrict the main survey download to. |
| `archival_projects.list` | download (`--archival`) | HST proposal IDs to whole-project-scan, one per line. |
| `archival_bad_obs.list` | download (`--archival`) | Observation-ID substrings to exclude from archival downloads (both phases). |
| `archival_targets.list` | download (`--archival`) | Optional observation-ID substrings to restrict the whole-project scan to; ignored if empty. |
| `archival_name_map.list` | download (`--archival`) | `MAST_NAME  HCNS_NAME` pairs (whitespace-separated, uppercase) for targets whose MAST target name differs from the canonical HCNS name. Auto-extended by the tsv-driven download phase. |
| `HCNS_Archival_Dwarfs.tsv` | download (`--archival`) | Master table of archival dwarf candidates with explicit per-target observation codes; **not tracked in git**, per-machine. Columns: `HCNS_Name`, `Other_Names`, `Codes (Observation ID)` (comma-separated), `Project#`, `Proposal_ID`, `ACS/WFC`, `WFC3/UVIS`, `Reason_Skipped`, `RA`, `Dec`. Rows with a blank `Codes` column are ignored. |

All observation-ID list files use the same matching convention: each line is
a substring checked (case-insensitively) against each MAST observation's
`dataURL`. Empty files are valid (no exclusion/restriction applied).

## Development / test scripts

Not tracked in git — local scratch scripts for one-off comparisons:
- `test_dolphot_versions.py` — runs the full pipeline independently under
  two different dolphot binary versions (paths configured via
  `DOLPHOT_BASE` near the top of the file) on a small target list, for
  comparing output.
- `test_hotpixel.py` — compares hot-pixel removal options for the RGB
  composite images on a single target.

## Setup notes

Before first use, each list file above should exist (even if empty) in this
directory. Scripts assume they are run with the working directory set to
`HCNS_data_reduction/` (all paths are relative to `os.getcwd()`).
