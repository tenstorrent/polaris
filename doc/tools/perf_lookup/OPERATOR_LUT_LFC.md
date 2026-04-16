# Operator performance LUTs via Large File Cache (LFC)

Package YAML under `config/` specifies **tt-perf master** lookup tables using `lfc://` paths for automatic download and caching:

| Architecture package | Config key | LFC Path |
|----------------------|------------|----------|
| Wormhole `n150` (`config/tt_wh.yaml`) | `operator_lookup_file` | `lfc://hlm-lut/whb0_n150_lut.yaml` |
| Blackhole `p100a` (`config/tt_bh.yaml`) | `operator_lookup_file` | `lfc://hlm-lut/bh_p100_lut.yaml` |

When using `lfc://` paths, files are **automatically downloaded** from LFC on first use and cached locally under `__ext/hlm-lut/`. The cache is checked for freshness (1 week) and re-downloaded if stale.

**Prerequisites for automatic download:**
- (Developers outside CI) **Tailscale** VPN connection to reach the LFC server
- (CI) Runs in cluster with direct LFC access

If LFC is unavailable and no cached file exists, the simulator will log a warning and continue without the performance lookup (maintaining backward compatibility).

## Manual Download (Optional)

Manual download is only needed for offline access or troubleshooting.

**Prerequisites:**
- Same requirements as [lfc_downloader_user_guide.md](../ci/lfc_downloader_user_guide.md): `wget`, and (for developers outside CI) **Tailscale** to reach the external LFC host
- Run commands from the **repository root**

On LFC, these LUTs live under a tree that **starts at `hlm-lut/`** (first segment after the downloader's `simulators-ai-perf` base URL; see [lfc_downloader_user_guide.md](../ci/lfc_downloader_user_guide.md)).

Sync to `__ext/hlm-lut/` to match the automatic cache layout:

```bash
./tools/ci/lfc_downloader.sh -v hlm-lut/ __ext/hlm-lut/
```

After a successful sync you should have at least:

- `__ext/hlm-lut/whb0_n150_lut.yaml`
- `__ext/hlm-lut/bh_p100_lut.yaml`

(Additional files may appear if the `hlm-lut` bundle grows; the simulator only loads the path named in package YAML.)

### Dry run

```bash
./tools/ci/lfc_downloader.sh -n -v hlm-lut/ __ext/hlm-lut/
```

### If LFC stores a single archive instead

If `hlm-lut` is shipped as a `.tar.gz` on the server, download and extract from the repo root:

```bash
./tools/ci/lfc_downloader.sh --type file --extract hlm-lut/<archive>.tar.gz __ext/hlm-lut/
```

The archive layout must place `whb0_n150_lut.yaml` and `bh_p100_lut.yaml` under `__ext/hlm-lut/`.

## Wire format

The normative YAML shape is documented in [YAML_MASTER_FORMAT.md](../../YAML_MASTER_FORMAT.md). Runtime behavior and key bridging are summarized in [LOOKUP_TABLE_MASTER.md](LOOKUP_TABLE_MASTER.md).

## Tests

Tests that need a LUT on disk assume an `lfc://` path is configured and can be automatically downloaded (cached under `__ext/hlm-lut/`), or you have manually downloaded files to `__ext/hlm-lut/`. They skip or fail if the file cannot be resolved.

## See also

- [lfc_downloader_user_guide.md](../ci/lfc_downloader_user_guide.md) — script options, CI vs developer mode, troubleshooting.
- [large_file_cache_usage.md](../ci/large_file_cache_usage.md) — general LFC usage patterns for this repo.
