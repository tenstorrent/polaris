# Introduction to Polaris

*A guided tour of what Polaris is, why it exists, and how its pieces fit together.*

This document is the **starting point**. It gives you the conceptual map at intro
altitude and then points you to the deeper references — [overview.md](overview.md)
for internals, [user_guide.md](user_guide.md) for hands-on usage — once you know
where you want to go.

---

## 1. What is Polaris, and why does it exist?

**Polaris** (the Python package is named `ttsim`) is a **high-level, roofline
performance simulator** for AI workloads running on Tenstorrent hardware (the
Wormhole and Blackhole families). You give it a model and an architecture spec; it
tells you how that model would perform on that chip — latency, cycles,
utilization, bandwidth — **without needing physical silicon**.

Two characterizations frame everything that follows:

- **It is a roofline simulator.** Each operation's achievable performance is
  bounded by the hardware's peak compute throughput and peak memory bandwidth, so
  Polaris reasons about whether work is **compute-bound or memory-bound** rather
  than tracking microarchitectural detail cycle by cycle.
- **It simulates entire workloads, not single operators.** The unit of interest
  is the whole model's operation graph, run end to end — that is what yields
  workload-level latency and utilization, not an isolated per-op microbenchmark.

The problem it solves: performance questions arrive long before hardware is
available or plentiful. *"How fast would this transformer run on Blackhole at
batch 32?"* *"Which layers are memory-bound?"* *"Is this fusion worth it?"*
Answering those on real silicon is slow, hardware-constrained, and impossible for
chips that don't exist yet. Polaris answers them analytically in seconds on a
laptop.

**How it differs from other approaches:**

- **vs. running on silicon** — no hardware required; you can model unreleased
  chips by editing a YAML spec, and you get per-op visibility that's hard to
  extract from real runs.
- **vs. cycle-accurate simulators** — Polaris is deliberately *high-level*. It
  models operations analytically (with optional silicon-measured lookup tables
  for accuracy) rather than simulating every clock cycle, trading some fidelity
  for enormous speed.

**Design goals:** fast turnaround, per-operation insight, and — via the TTNN shim
(§2) — the ability to run the *same workload code* both on real hardware and in
the simulator. **Non-goals:**
Polaris is a **single-chip** simulator — one run models one package. It does not
model multi-chip topology, collectives, or fabric. Multi-chip workloads require
per-chip mapping plus external cross-chip modeling.

**Current fidelity (a different kind of limitation).** The single-chip boundary
above is a deliberate design choice. Separately, the roofline model is still
maturing, so **not every architectural element is modeled yet** — for example,
on-core **L1 (SRAM) is not currently modeled**. This is a present gap rather than a
design choice, and the set of modeled elements is expected to grow as the
analytical model is refined (see the LUT-driven insight in §7).

---

## 2. Polaris's signature innovation: the TTNN shim

Tenstorrent's production software stack exposes a Python API called **`ttnn`**.
Real workloads are written against it — `ttnn.matmul(...)`, `ttnn.conv2d(...)`,
and so on — and those calls execute kernels on the device.

Polaris's signature innovation is the **TTNN shim**: a **drop-in replacement for
the `ttnn` library** that presents the *exact same API surface* but, instead of
executing kernels, **records each call as a node in a graph**. A workload written
for real hardware runs against the shim unchanged; nothing touches silicon.

This enables an important design goal — **dual-mode operation**. The point is
that a TTNN-shim workload runs with *almost identical* code on hardware and in
Polaris, so a Polaris projection can be compared against a real hardware run
without an apples-vs-oranges mismatch in what was actually executed. The same
workload file runs:

- **On hardware** — imports the real `ttnn`, executes for real.
- **In Polaris** — imports the shim, builds a graph the simulator can analyze.

What makes that comparison trustworthy — and not just a similar-looking script —
is a further design aim of the shim: to emit a **hardware-aligned operator-instance
sequence**. The intent is a 1:1 correspondence in which the *N*-th operator
instance in Polaris's output lines up with the *N*-th in the hardware output, at
matching **op type**, with the same relevant **output attributes** (shape, and —
on a per-op-type basis — dtype, layout, memory config). Op *names* need not match.
This is a goal rather than a guarantee: for the shims and workloads on `main` it is
mostly achieved, with minor gaps possible.

That alignment is what pays off downstream — twice over. It makes per-operator
comparison against hardware meaningful (each op is matched to its real counterpart
rather than compared coincidentally), and it makes the same per-op hardware
information available as a substrate for **improving the analytical model** — see
the lookup-table discussion in §7.

The switch is made by an environment variable (`IRD_ARCH_NAME`): present means
"use real `ttnn`," absent means "use the shim." Within the shim, the
`ExecutionMode` constants class (`TRACK_ONLY`, `EXECUTE`, `EXECUTE_AND_TRACK`)
controls whether a call is merely recorded, actually executed, or both.

> **Important scope note:** the shim is the mechanism of *one* of Polaris's three
> frontends (the TTNN one). It is the most novel and the primary path, but — as
> the next two sections make clear — it is **not** how every workload enters
> Polaris. Two other frontends build the same graph by entirely different means.

The shim lives under [ttsim/front/ttnn/](../ttsim/front/ttnn/); its architecture
is documented in depth in [ttnn_shims_README.md](ttnn_shims_README.md).

---

## 3. The conceptual flow: three paths in, one path out

The shim is not the whole story of how workloads enter Polaris. Polaris has
**three frontends that build the graph in three different ways**, and they
**reconverge** on a single, frontend-agnostic backend. Only the TTNN frontend
uses a shim — keep that in mind as you read the flow below.

```
FRONTENDS  — divergent: they differ in HOW the graph is built
┌───────────────────────────────────────────────────────────────┐
│  TTSim Functional   Python Module code ─► builder API ────────┐ │
│                     (creates SimOps directly — no shim)       │ │
│                                                               │ │
│  ONNX               model.onnx file ─► parser ───────────────►│ │
│                     (translates ONNX nodes into SimOps)       │ │
│                                                               │ │
│  TTNN   ┌── workload calls ttnn.add / ttnn.matmul / ...        │ │
│         └─► SHIM intercepts ─► accumulates in a shim Device ──►│ │
│             (drop-in ttnn API; nothing runs on hardware)      │ │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼   all three emit the SAME structure
                     ┌────────────────────────┐
                     │   WorkloadGraph         │   NetworkX DAG of
                     │   SimOp nodes,          │   SimOp / SimTensor
                     │   SimTensor edges       │
                     └────────────────────────┘
                                │
   SHARED BACKEND  — frontend-agnostic from here on
                                ▼
                wl2archmapping post-processing
                (op fusion · dtype assignment · op removal
                 · compute-pipe assignment)
                                │
                                ▼
                Device.execute_graph()
                (walks the DAG in topological order · applies the
                 analytical perf model · optionally substitutes
                 silicon-measured values from a LUT)
                                │
                                ▼
                      Stats  (per-op + rollup CSV, metrics)
```

The key idea: **the frontend chooses how the DAG is *authored*, but once a
`WorkloadGraph` exists, the rest of the pipeline is identical regardless of where
it came from.** `Device` has no knowledge of which frontend produced the graph it
executes.

---

## 4. Architecture layers

Polaris is organized as a stack. From entry point down:

| Layer | Location | Role |
|-------|----------|------|
| **CLI entry point** | [polaris.py](../polaris.py) | Parses flags, selects frontend per workload, drives the run |
| **Frontends** | [ttsim/front/](../ttsim/front/) | Turn a workload into a `WorkloadGraph` (functional / onnx / ttnn) |
| **Graph** | [ttsim/graph/](../ttsim/graph/) | `WorkloadGraph` — the NetworkX DAG that all frontends produce |
| **Ops** | [ttsim/ops/](../ttsim/ops/) | `SimOp` (a node) and `SimTensor` (an edge); shape inference under `ops/desc/` |
| **Backend** | [ttsim/back/](../ttsim/back/) | `Device` — executes the graph in topo order and applies perf models |
| **Config** | [ttsim/config/](../ttsim/config/) + [config/](../config/) | Architecture specs, workload registry, graph-mapping transforms |
| **Stats** | [ttsim/stats/](../ttsim/stats/) | Collects and emits metrics (per-op and rolled-up CSV) |

The classes you'll meet most often:

| Class | File | What it is |
|-------|------|------------|
| `WorkloadGraph` | [ttsim/graph/wl_graph.py](../ttsim/graph/wl_graph.py) | The DAG of ops and tensors |
| `SimOp` | [ttsim/ops/op.py](../ttsim/ops/op.py) | A single operation node |
| `SimTensor` | [ttsim/ops/tensor.py](../ttsim/ops/tensor.py) | A tensor edge between ops |
| `Device` | [ttsim/back/device.py](../ttsim/back/device.py) | The execution/simulation engine |

Two configuration inputs deserve a specific mention because they shape every run:

- **Architecture spec** — [config/tt_wh.yaml](../config/tt_wh.yaml) (Wormhole),
  [config/tt_bh.yaml](../config/tt_bh.yaml) (Blackhole). Core counts, SRAM,
  bandwidths — the hardware you're modeling.
- **Graph-mapping spec** — [config/wl2archmapping.yaml](../config/wl2archmapping.yaml).
  The post-processing rules applied after the graph is built: op fusion, dtype
  assignment, op removal, and compute-pipe assignment. This is the
  "shared backend" box's first step in the diagram above.

---

## 5. The three frontends — which one, and how it works

All three produce the same `WorkloadGraph`. They differ in **what the input is**
and **how the graph gets built**. Each workload declares its frontend in
[config/all_workloads.yaml](../config/all_workloads.yaml).

| Frontend | Input | How the graph is built | Use it when… |
|----------|-------|------------------------|--------------|
| **TTNN shim** | Python code calling `ttnn.*` | **Shim intercepts** each API call and accumulates `SimOp`s in a shim `Device` | You want the same code to run on silicon *and* in Polaris (dual-mode), or you're modeling a real `ttnn` workload |
| **TTSim Functional** | Python code using the `Module` API | A **builder API** creates `SimOp`s **directly** as you compose the model | You're authoring a model natively for Polaris and don't need hardware execution |
| **ONNX** | A serialized `.onnx` model file | A **parser** reads the file and **translates ONNX nodes** into `SimOp`s | You already have a model exported to ONNX and just want to simulate it |

The distinction the diagram captured is the crucial one:

- **TTNN is the only frontend that shims/intercepts.** The workload thinks it's
  calling the real `ttnn`; Polaris quietly records instead of executing.
- **Functional builds directly.** There's no interception — your code explicitly
  drives a graph builder, and `SimOp`s are created as you go. See
  [functional.md](functional.md) and, if porting from PyTorch,
  [torch2ttsim.md](torch2ttsim.md).
- **ONNX parses a file.** No code runs at all in the modeling sense; the graph is
  a translation of the ONNX protobuf.

After any of these, the graph is identical in type and structure, and the backend
treats it identically.

---

## 6. A run, end to end

At the intro level, one command is enough to see the shape of a run. (For the full
flag reference and how to read every output, go to
[user_guide.md](user_guide.md) and the top-level [README.md](../README.md).)

```bash
python polaris.py \
  -w config/all_workloads.yaml \
  -a config/tt_wh.yaml \
  -m config/wl2archmapping.yaml \
  --filterarch n150 --filterwli gpt_nano \
  --study my_study -o __output/
```

Reading it: `-w` is the workload registry, `-a` is the architecture spec (here a
Wormhole n150), `-m` is the graph-mapping spec, `--filterarch`/`--filterwli` narrow
the run to one SKU and one workload, and `-o` is where results land. Internally,
this selects the workload's frontend, builds the `WorkloadGraph`, applies
`wl2archmapping`, executes it on the `Device`, and writes stats to `__output/`.
Use `config/tt_bh.yaml` instead of `tt_wh.yaml` to model Blackhole.

---

## 7. Lookup tables and perf accuracy

The analytical perf model is fast but approximate. To raise fidelity, Polaris
supports **performance lookup tables (LUTs)** — YAML files keyed by
`op_type + shapes + layout + dtype` (extended in later schemas with per-op variant
keys) that hold **profiled silicon measurements**. A LUT is attached to the
**architecture spec**: the SKU's config carries an `operator_lookup_file`
attribute pointing at its LUT (for example
[config/tt_wh.yaml](../config/tt_wh.yaml) wires one to the n150). When a LUT is
loaded, the backend **substitutes the measured value for the analytical estimate**
on any op whose key hits the table — the "LUT substitution" step in the flow
diagram.

**What LUTs are for — and what they are not.** LUTs are introduced with several
deliberate aims, and it helps to hold all of them at once:

- **They generate modeling insight.** Deciding *what belongs in the key* is a
  hypothesis about which factors drive an operator's performance — layout,
  residence, and how those shape memory traffic, for example. Evolving the key,
  and reading the per-op gap between the analytical estimate and the measured
  value, reveals both *which* factors matter and *how* — i.e. exactly where the
  analytical model is weak. That knowledge is **intended to feed back into the
  analytical model**.
- **They give Polaris a built-in correlation capability.** With a LUT in hand,
  Polaris can compare its own projections against a hardware reference from inside
  the tool (hit rate + gap), rather than relying solely on an external pipeline.
- **They are a principled scaffold, not a fudge.** A hit should come from sound
  modeling that legitimately matches the profiled operator — not from an
  unprincipled fallback contrived to make the numbers line up.

> **A note on how to read this:** the LUT is a deliberate **intermediate step**
> toward a better-correlated analytical model — not Polaris's definition of
> accuracy, and not an end in itself. The firm intent is for the analytical model
> to be good enough on its own; whether the LUT is ultimately retained for a
> residual set of hard-to-model ops or dropped entirely is an open question. Read
> LUTs as a research instrument and a stepping stone — neither a throwaway hack
> nor the authoritative answer.

For the wire format and authoring workflow, see
[YAML_MASTER_FORMAT.md](YAML_MASTER_FORMAT.md) and the LUT tooling docs under
`doc/tools/perf_lookup/`.

---

## 8. Glossary and where to go next

**Glossary**

- **`ttsim`** — the Python package name for Polaris.
- **Roofline model** — a performance model that bounds an operation by the
  hardware's peak compute throughput and peak memory bandwidth, classifying it as
  compute-bound or memory-bound; Polaris's analytical model is roofline-based.
- **Shim** — Polaris's drop-in replacement for the real `ttnn` library; records
  calls instead of executing them. The mechanism behind the TTNN frontend only.
- **Frontend** — one of three ways a workload becomes a graph: TTNN shim, TTSim
  Functional, or ONNX.
- **`SimOp` / `SimTensor`** — a node (operation) / an edge (tensor) in the graph.
- **`WorkloadGraph`** — the NetworkX DAG of `SimOp`s and `SimTensor`s that every
  frontend produces and the backend consumes.
- **`Device`** — the backend engine that walks the graph in topological order and
  applies the perf model.
- **`wl2archmapping`** — the post-build transform pass (fusion, dtype, removal,
  compute-pipe assignment).
- **Compute pipe** — the hardware execution path a given op is assigned to during
  mapping.
- **LUT (lookup table)** — YAML of silicon-measured op performance that overrides
  analytical estimates for matching ops.
- **SKU** — a specific hardware variant (e.g. n150/n300 for Wormhole;
  p100a/p150a/p150b for Blackhole).
- **Dual-mode** — the property that one workload file runs both on real hardware
  (real `ttnn`) and in Polaris (the shim).

**Suggested reading path**

1. **This document** — the conceptual map (you're here).
2. **[README.md](../README.md)** — install and quick-start commands.
3. **[user_guide.md](user_guide.md)** — full CLI usage, config files, output
   formats, debugging.
4. **[overview.md](overview.md)** — the internals reference: `WorkloadGraph`,
   `SimOp`, `SimTensor`, `Device`, and the frontends in detail.
5. Depending on your goal:
   - Authoring a native model → [functional.md](functional.md),
     [torch2ttsim.md](torch2ttsim.md)
   - Working with the TTNN shim → [ttnn_shims_README.md](ttnn_shims_README.md),
     [TTNN_WORKLOAD_FLOW.md](TTNN_WORKLOAD_FLOW.md)
   - Shape inference rules → [shape_inference.md](shape_inference.md)
   - Perf LUTs → [YAML_MASTER_FORMAT.md](YAML_MASTER_FORMAT.md)
