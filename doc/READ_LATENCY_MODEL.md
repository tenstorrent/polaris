# DRAM Read-Latency Model (`--enable_dm_latency`)

Polaris' default memory model is flat bytes/bandwidth: a read costs `inBytes / effective_bandwidth`. That is a good approximation when an op streams enough data to saturate the memory system, and a bad one when it does not. A single 2 KB tile read from DRAM costs ~570 NoC cycles on Blackhole no matter how much bandwidth is idle, because the cost is a fixed DRAM access bucket plus a NoC round-trip plus a per-read issue cost — none of which shrink with the byte count.

This model closes that gap. When enabled, read cycles become

```
mem_rd_cycles = max(bandwidth_limited_cycles, predicted_read_latency)
```

**Small transactions become latency-bound; large transactions stay bandwidth-bound and are numerically unchanged.** That `max` is the entire integration, and the rest of this document explains the formula feeding it, where its constants come from, and what is deliberately not modeled.

Off by default. Enable with `polaris.py --enable_dm_latency` (or `enable_dm_latency: true` in a run config).

## Where things live

| Concern | Location |
|---|---|
| The analytical model (pure function, no Polaris types) | `ttsim/back/read_latency.py` |
| Calibrated HW constants (single source of truth) | `config/tt_bh.yaml`, memory block `read_latency:` |
| Constant schema / validation | `MemoryReadLatencyModel` in `ttsim/config/simconfig.py` |
| Arch+workload derivation and timing integration | `Device._dm_read_latency_devclk` / `Device.execute_op` in `ttsim/back/device.py` |
| Calibration tests (microbenchmark ground truth) | `tests/test_back/test_read_latency.py` |
| Layer-1 unary cross-check | `tests/test_back/test_read_latency_layer1.py` |
| Integration tests (`max` semantics) | `tests/test_back/test_dm_latency.py` |

Upstream source: the O2O "Read Latency Modeling" Confluence page (2433646619), issue #472.

## The model

`predict_read_latency(N, Q, hops, num_channels, cfg)` predicts the latency, in NoC/fabric clock cycles (fclk), of one Tensix core issuing `Q` `noc_async_read` transactions of `N` bytes each from DRAM, followed by a barrier. It is the larger of two arms — whichever constraint binds:

```
base            = Tdram + 2 * hops * Chop

issue_bound     = delta_issue * Q  +  base  +  Tdetect  +  N / B_channel
transport_bound = min(Q, n*) * delta_issue  +  base  +  Q * N / B_eff

B_eff = min(min(Q, num_channels) * B_channel, noc_inbound_bpc)
n*    = ceil(noc_inbound_bpc / B_channel)
```

The **issue arm** dominates for small `N` with a deep queue: the core serializes on issuing requests, and the barrier tail (`Tdetect`) is exposed. The **transport arm** dominates for large `N * Q`: reads pipeline, requests overlap, and the per-core NoC inbound link is the ceiling. `n*` is the number of DRAM channels needed to saturate that link, so beyond it extra channels buy nothing.

Note the transport arm omits `Tdetect`. That is intentional and matches the calibrated source formula — in the streaming regime the barrier clears behind in-flight data.

## Why `max`, not `bandwidth + latency`

An earlier iteration of this work (see below) integrated additively: it decomposed the prediction into an "exposed" part (fill + drain + serial issue) and a "hideable" part (streaming delivery), then added only the exposed part on top of the bandwidth cost. That is defensible physics, but it cost a `ReadLatencyBreakdown` dataclass, a parallel set of `*_breakdown` functions, a shadow `mem_rd_cycles_bw_fractional` field on every op, a duplicate aggregation loop, and a special case in the LUT path — all of it existing purely to avoid double-counting the streaming bytes that the additive form counts twice.

`max` needs none of that, because the transport arm already contains the streaming term. There is nothing to subtract.

The trade-off is real and worth knowing before anyone "fixes" this: `max` hides the fill latency on ops that are bandwidth-bound, where physically you still wait for the first byte. Two reasons that is acceptable at this level of fidelity:

1. Whether the fill is exposed depends on read/compute pipelining that Polaris does not model per-op.
2. `get_exec_stats` already computes `ideal_cycles = max(compute_cycles, mem_cycles)`, so the memory term is discarded outright whenever compute dominates. Adding a fixed head below that `max` would frequently be invisible anyway.

If a future calibration shows the fill is genuinely exposed on bandwidth-bound ops, reintroduce it as an explicit additive term with its own evidence — do not resurrect the exposed/hideable decomposition just to make the bandwidth self-check pass.

## Calibration constants

All constants live in the arch YAML and are **required** when the feature is enabled; there is no Python-side fallback. Enabling `--enable_dm_latency` on an arch whose memory block has no `read_latency:` section raises a `ValueError` naming the device. Today only `config/tt_bh.yaml` (the `gddr6_bh` memory IP) carries one — `config/tt_wh.yaml` does not, so the flag is Blackhole-only. The check happens per device during construction, so pairing the flag with a multi-arch spec such as `config/all_archs.yaml` aborts the sweep partway rather than skipping the unsupported devices.

The rate and cost fields are schema-constrained positive, because the model divides by `b_channel_bpc` and `noc_inbound_bpc`: a mistyped `0` would otherwise surface as a bare `ZeroDivisionError` with no indication of which field was wrong, and a negative constant would yield a silently negative latency that `max()` quietly absorbs.

```yaml
read_latency:
    tdram_cyc            : 375.0   # DRAM access bucket (row activate + CAS + burst)
    tdetect_cyc          : 10.0    # barrier-clear tail
    noc_inbound_bpc      : 62.0    # per-core NoC inbound ceiling (bytes / fclk cycle)
    delta_issue_cyc      : 38.0    # per-read issue cost (cyc/read)
    chop_cyc_per_hop     : 8.0     # NoC cost per hop (cyc)
    b_channel_bpc        : 24.0    # single-channel effective rate (bytes / fclk cycle)
    default_hops         : 4       # gating-channel hop distance (h_gate)
```

Two values are **not** in this block:

- `num_dram_channels` comes from the memory IP group's `num_units` on the specific package instance — 7 on p100a, 8 on p150a/p150b, from the same shared constants. Note this has **no effect on any shipped Blackhole config**: `B_eff` is capped at `noc_inbound_bpc = 62` and `n* = ceil(62/24) = 3`, so every channel count ≥ 3 gives a bit-identical prediction. The parameter only matters for a hypothetical 1- or 2-channel part.
- `fclk_mhz` is an optional field on the block. When absent (the current state) fclk defaults to the matrix compute clock, which is correct on Blackhole because the NoC is tied to the AI clock. Both are 1350 MHz on p100a, so the fclk→devclk conversion is currently a no-op — but the conversion is applied unconditionally, so a future arch with a decoupled NoC clock only needs the YAML field.

## Arch and workload derivation

`Device._dm_read_latency_devclk` maps the simulator's world onto the model's four scalars. This is the only place that mapping happens:

| Model input | Source | Detail |
|---|---|---|
| `N` (page bytes) | workload | One 32×32 tile at the op's precision — `TILE_ELEMS * bpe`. Tile-paged reads are the DRAM-interleaved default. |
| `Q` (queue depth) | workload + arch | `ceil(ceil(inBytes / N) / num_cores)`, floored at 1. Reads fan across the whole compute grid. |
| `num_channels` | arch | Memory IP `num_units`. |
| `hops` | arch (constant) | `default_hops`; per-op hop distance is not modeled. |

Both divisions round **up**. A partial tile still costs a whole transaction, and a partial round across the grid still leaves some core issuing one more read than its peers — and that core is what the barrier waits for. Earlier revisions used floor for `Q` and `round` for the page count; both under-counted. `round` in particular could discard exactly the partial page the `ceil` rationale exists to capture: at `N=1024`, `inBytes=123392` rounds to 120 pages (`Q=1`, 530 cyc) where it should ceil to 121 (`Q=2`, 568 cyc).

## Validation

`test_read_latency.py` checks the model against 45 rows of Blackhole ground truth from tt-metal's `DRAM Interleaved Page Read Numbers.csv` (transaction sizes 64 B – 16 KB × queue depths 1 – 256, riscv_1). The rows are embedded in the test so it is hermetic, and the constants are read from the shipped YAML so the test validates the *shipped* calibration rather than a duplicated copy.

Current agreement: **mean relative error 1.99%, worst case 10.0%** (at N=256, Q=1). Test gates are ≤15% per row and ≤5% mean. The suite also asserts monotonicity in both `N` and `Q`, that effective bandwidth saturates near the NoC inbound ceiling, and that channel counts at or above `n*` are indistinguishable while a 1-channel part is strictly slower.

Fit gates alone are a weak guard on *structure*: a 45-row fit at 15%/5% tolerates several formula mutations that are physically wrong but numerically close. `test_issue_arm_closed_form` and `test_transport_arm_closed_form` therefore assert exact closed-form equality at one point in each regime, chosen so that between them they pin `Tdetect`'s presence in the issue arm and absence from the transport arm, the `delta_issue` slope, the `ceil` in `n*`, and the issue term inside the transport arm. Mutating any of those fails a test; before these existed, only swapping the final `max` for `min` was caught.

One known-uncovered mutation: dropping the `min(Q, ·)` factor from `B_eff` changes nothing under the shipped constants (see the comment at that line), so no test can distinguish it.

`test_read_latency_layer1.py` is a separate cross-check against the conf page's Layer-1 table (single core, single channel, Q=1, 64 B – 512 KB). It carries a *local* reference implementation of the unary closed form, which models a >32 KB pipelined dual-rate flit delivery regime that the production two-arm model deliberately does not. Polaris never calls it. It lives in the test, with its five constants inline, so that the arch YAML and the production dataclass are not burdened with fields nothing reads. Do not promote it back into `read_latency.py` without a production caller.

One known discrepancy, documented in that test: the conf page's 128 KB "Predicted" cell reads 4714, but the page's own formula yields 4679.68 and the measured value is 4679. Treated as a typo on the page.

## Deliberately out of scope

**L1 and sharded reads.** Every read is priced with the DRAM-interleaved calibration, including inputs that are actually L1-resident. This over-charges L1 reads. A regime registry keyed on each input's `MemoryConfig` canonical tag (`DRAM_INTERLEAVED`, `L1_BLOCK_SHARDED`, …) was built and then removed, because the only non-DRAM parameter set available was an uncalibrated guess, and shipping a code path plus a schema layer plus a validator to serve numbers nobody should trust is a bad trade. The work is preserved in branch history — `bsheikh/dm-latency-interleaved-reads-A`, commits *"regime-aware read latency + exposed/hideable breakdown"* and *"add `--dm_latency_mode apply`"* — and should be revived when an L1-sharded read microbenchmark exists to calibrate against. Reviving it means restoring `ReadSourceDescriptor`, the per-input descriptor builder, and the nested `regimes:` YAML schema.

**Multi-input serialization.** The prediction is derived from the op's aggregate `inBytes`, not per-input. A matmul reading two operands shares one NoC inbound link, and this model does not reason about that sharing.

**Also not modeled:** NoC congestion and VC contention, DRAM row-miss and refresh penalties, per-op hop distance, and writes. The model is read-only; `mem_wr_cycles` is untouched.

## Aggregate bandwidth self-check

`get_exec_stats` validates that total read cycles are consistent with `tot_inBytes / effective_bandwidth`. Latency-bound reads legitimately exceed that, so when the model is enabled the over-count direction is no longer an error:

```python
too_low  = tot_mem_rd_cycles < expected_cycles - 1
too_high = tot_mem_rd_cycles > expected_cycles + 1 and not self.enable_dm_latency
```

The under-count direction remains a hard error in both modes — nothing in this feature can make reads cheaper, so an under-count still means broken traffic accounting.

## Observed impact

| Workload | Read cycles | Total cycles |
|---|---|---|
| RESNET50 (bs=1, p100a, all 3 instances) | +0.00% | +0.00% |
| llama3_decode (bs=1, p100a) | +0.46% | +0.37% |

ResNet's zero is instructive, and it is **not** because every op is bandwidth-bound. The latency arm actually wins on 33 of the 143 ops in `rn50_224x224` (mostly `BatchNormalization`, plus a few `Relu`/`Reshape`), raising each of them to a flat 530 read cycles from anywhere between 8 and 409. The totals are unchanged only because *all 33 are ops that fusion eliminates* — they end up `fused=True` with `ideal_cycles=0.0`, so they contribute nothing to the aggregate. The convolutions that survive fusion do move enough bytes to be genuinely bandwidth-bound.

That makes the zero a fusion artifact, not a property of the model, and it will not survive a fusion-spec change. With `--disable-fusion` the same workloads move:

| Workload | Read cycles | Total cycles |
|---|---|---|
| `rn50_224x224` | +3.53% | +2.24% |
| `rn50_b1_hd` | +0.10% | +0.05% |
| `rn50_b1_uhd` | +0.01% | +0.00% |

The trend across those three is the model behaving as designed: the latency arm binds on 23% / 4% / 1% of ops as the input resolution grows, because larger activations push more ops past the crossover. Decode workloads, with small per-token activations, are where the arm binds on ops that fusion keeps.

A near-zero delta is therefore an expected result for large-tensor workloads and is not evidence the flag is broken. To confirm the model is running, look for `DM-READ-LAT` at `DEBUG` level, which logs `N`, `Q`, channel count, and the predicted latency for every op. It is at `DEBUG` rather than `INFO` because it is one line per op per device, and because it is also emitted for ops that later take a LUT hit and have their memory cycles zeroed — where the logged prediction is not what the op is ultimately charged.

## Extending

- **New arch:** add a `read_latency:` block to its memory IP. Every field is required; `extra='forbid'` will reject typos.
- **Recalibrating:** change the YAML only. `tests/test_back/conftest.py` builds the test config from the same YAML, so the microbenchmark test will immediately tell you whether the new constants still fit the 45 ground-truth rows.
- **New regime (L1, sharded, …):** see "Deliberately out of scope" — get the microbenchmark first, then restore the regime schema from branch history.
