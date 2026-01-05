# Operator performance LUTs via Large File Cache (LFC)

Package YAML under `config/` can point simulators at **tt-perf master** lookup tables, for example:

| Architecture package | Config key | File (repo-relative) |
|----------------------|------------|----------------------|
| Wormhole `n150` (`config/tt_wh.yaml`) | `operator_lookup_file` | `__ext/perf_lookup/whb0_n150_lut.yaml` |
| Blackhole `p100a` (`config/tt_bh.yaml`) | `operator_lookup_file` | `__ext/perf_lookup/bh_p100_lut.yaml` |

Those YAML files are **large, curated artifacts** and are **not checked into this repository**. Fetch them from LFC using `tools/ci/lfc_downloader.sh`.

## Prerequisites

- Same requirements as [lfc_downloader_user_guide.md](../ci/lfc_downloader_user_guide.md): `wget`, and (for developers outside CI) **Tailscale** to reach the external LFC host unless you have direct access.
- Run commands from the **repository root** so paths match `config/*.yaml`.

## Download (recommended)

On LFC, these LUTs live under a tree that **starts at `hlm-lut/`** (first segment after the downloader’s `simulators-ai-perf` base URL; see [lfc_downloader_user_guide.md](../ci/lfc_downloader_user_guide.md)). Pass that as the script’s **`server_path`**, e.g. `hlm-lut/` to sync the whole bundle.

Sync that directory into `__ext/perf_lookup/` so the filenames match `operator_lookup_file`:

```bash
./tools/ci/lfc_downloader.sh -v hlm-lut/ __ext/perf_lookup/
```

After a successful sync you should have at least:

- `__ext/perf_lookup/whb0_n150_lut.yaml`
- `__ext/perf_lookup/bh_p100_lut.yaml`

(Additional files may appear if the `hlm-lut` bundle grows; the simulator only loads the path named in package YAML.)

### Dry run

```bash
./tools/ci/lfc_downloader.sh -n -v hlm-lut/ __ext/perf_lookup/
```

### If LFC stores a single archive instead

If `hlm-lut` is shipped as a `.tar.gz` on the server, download and extract from the repo root so contents land under `__ext/perf_lookup/` (adjust the archive name to match what LFC hosts):

```bash
./tools/ci/lfc_downloader.sh --type file --extract hlm-lut/<archive>.tar.gz
```

The archive layout must place `whb0_n150_lut.yaml` and `bh_p100_lut.yaml` under `__ext/perf_lookup/` after extraction, or you must move them to match `operator_lookup_file` in `config/tt_wh.yaml` and `config/tt_bh.yaml`.

## Wire format

The normative YAML shape is documented in [YAML_MASTER_FORMAT.md](../../YAML_MASTER_FORMAT.md). Runtime behavior and key bridging are summarized in [LOOKUP_TABLE_MASTER.md](LOOKUP_TABLE_MASTER.md).

## Tests

Tests that need a LUT on disk (for example under `__ext/perf_lookup/`) assume you have already run the downloader; they skip or fail if the file is missing.

## See also

- [lfc_downloader_user_guide.md](../ci/lfc_downloader_user_guide.md) — script options, CI vs developer mode, troubleshooting.
- [large_file_cache_usage.md](../ci/large_file_cache_usage.md) — general LFC usage patterns for this repo.
