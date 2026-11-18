# QOBLib QUBO Reader — Architecture, Semantics, and Developer Guide

**Development snapshot:** 2026-11-16

**Scope:** single-instance `.qs` reading, interpretation, convention conversion, structural inspection, and output adapters.

**Location:** `my_code/qubo_reader/README.md` (run examples from the **QOBLIB repository root**).

> **Where we are:** The five modules below have been implemented. The adapter test file was reported to pass **13 tests** on the development machine. The complete multi-module test suite and the actual MQLib executable still require a coordinated end-to-end validation. This document records what the code **does**, what its tests **establish**, and what remains an **assumption**.
>
> **Next practical action after reviewing this README:** Preserve it in the repository, review the existing test suite, and make a Git checkpoint before starting Phase B (real-instance end-to-end validation). Bulk/directory loading is a later Phase C task.

## 1. Purpose: what problem are we solving?

The reader turns a **stored QOBLib QUBO instance** into several useful representations without silently deciding what its numbers mean.

A raw `.qs` file gives us a variable count, sparse coefficient entries, perhaps an objective offset, and comments. That does **not**, by itself, establish whether a stored off-diagonal number contributes once or twice to the objective, or whether the encoded objective should be minimized or maximized. The original model's declared direction can differ from the direction of the stored QUBO energy.

The design therefore separates five jobs:

1. **Parse** the file exactly as stored, validating its structure.
2. **Model** the raw sparse instance and its provenance without altering coefficients.
3. **Interpret** the instance under **explicit caller-supplied assumptions**.
4. **Convert** the interpreted objective to an explicit target matrix convention and optimization direction.
5. **Adapt** the converted result to the data structure a downstream consumer requires.

In parallel, the raw instance can be inspected **structurally** (edges, interaction dictionary, NetworkX support graph) without choosing a mathematical interpretation.

The reader is **not** a solver, a QUBO-to-Max-Cut reduction system, an automatic source-semantics detector, a multiprocessing scheduler, or a bulk directory loader.

### Architecture at a glance

```text
                         input: one .qs file
                                 |
                         qs.parse_qs(path)
                                 v
                           QSInstance
                                 |
                      model.from_qs(parsed)
                       or model.load_qs(path)
                                 v
                         QuboInstance (RAW)
                          /              \
                         /                \
             graph.py (side path)      interpret_qubo(...)
             structural inspection          |
               raw edges / raw graph        v
                                     InterpretedQubo
                                     polynomial E(x)
                                            |
                                     convert_qubo(...)
                                            v
                                       ConvertedQubo
                                     (chosen target form)
                                            |
                                       adapters.py
                            +---------------+---------------+
                            |               |               |
                       dimod.BQM      networkx.Graph     callable f(x)
                                                            |
                              also: MQLib temporary .qubo file
```

**Key rule:** `graph.py` uses a **raw** `QuboInstance`; the graph adapter in `adapters.py` uses an already **converted** `ConvertedQubo`. They serve different purposes and carry different coefficient meanings.

## 2. Terminology and mathematical contract

All variables are **binary**, `x_i in {0, 1}`, with **1-based labels** `1, ..., n`. All source and convention-layer coefficients use `decimal.Decimal`.

### Three different representations of the same input data

| Representation | What the coefficients mean | What is still undecided? |
|---|---|---|
| `QuboInstance` | Numbers stored in `.qs` | Meaning of off-diagonal coefficients and stored energy direction |
| `InterpretedQubo` | Full coefficients of the actual polynomial terms | Which target representation a consumer wants |
| `ConvertedQubo` | Matrix entries under a selected target convention and direction | Which Python structure/file/callable to serve |

**Interpreted polynomial:**

\[
E_{\mathrm{interpreted}}(x)
= c + \sum_i a_i x_i + \sum_{i<j} b_{ij}x_ix_j.
\]

The quadratic dictionary in `InterpretedQubo` stores the **full** `b_ij` polynomial coefficients, never half-coefficients.

**Matrix conventions:**

| Stored `(i, j, q)` with `i < j` | Polynomial contribution |
|---|---|
| `upper_triangular` | `q * x_i * x_j` |
| `symmetric` | `2 * q * x_i * x_j` because `Q[i,j] = Q[j,i] = q` in `x.T @ Q @ x` |

Diagonal entries always contribute **once**, because `x_i² = x_i` for binary `x_i`.

**Direction conversion:** Let `s = +1` when the interpreted and target directions agree, and `s = -1` otherwise. Then, for **every** binary assignment,

\[
E_{\mathrm{converted}}(x)=s\,E_{\mathrm{interpreted}}(x).
\]

A direction flip negates **all** linear terms, quadratic terms, and the **offset**. Thus `argmin E = argmax(-E)` (as sets of optimal assignments); the objective values themselves do not stay numerically equal.

**Offset rule:** An absent offset (`None`) is **not** the same as an explicit zero. `interpret_qubo` requires an explicit `offset_override` if the file did not supply one. Passing an override deliberately takes precedence over the file value. Avoid `float` for overrides; use `Decimal`, `int`, or decimal strings.

### One consistent running example

For clarity, each module below uses this tiny `.qs` file:

```text
# ObjectiveOffset 7
# Vars Non-zeros
4 2
1 1 -2
1 2 3
```

- `n = 4` declared variables, including variables 3 and 4 with no stored terms.
- `m = 2` nonzero stored entries: **one diagonal** and **one off-diagonal**.
- Raw off-diagonal stored coefficient is `3`.
- No unique polynomial energy is asserted **until** we select a source convention.

If we explicitly interpret the source as **upper-triangular, minimize**, the polynomial becomes:

\[
E(x)=7-2x_1+3x_1x_2.
\]

If instead we assume **symmetric, minimize**, that *same file* describes:

\[
E(x)=7-2x_1+6x_1x_2.
\]

Both interpretations are supported; only evidence external to the raw triples can justify which is intended for a specific dataset family.

## 3. Module-by-module reference

### 3.1 `qs.py` — strict `.qs` parsing (Layer 1)

**Position:** Entry from a file path. Used by `model.load_qs()`; can also be called independently.

**Public objects/functions:**
- `parse_qs(path) -> QSInstance`
- `QSInstance`: parsed `source`, `num_variables`, `declared_nonzero_count`, `offset`, `declared_direction`, `terms`, `comments`; computed `num_interactions`, `num_diagonal_terms`, `graph_density`
- `QSTerm(i, j, coefficient)`: one unchanged stored triplet
- `QSFormatError(ValueError)`: structural/format errors
- `main()`: small command-line inspection entry point

**Input:** One text `.qs` file, e.g. the running example. The expected structure is a preamble of `#` comments, a `# Vars Non-zeros` marker, a numeric `n m` line, and `m` sparse `i j coefficient` records.

**Output example (conceptual fields):**

```python
parsed = parse_qs(path)
parsed.num_variables          # 4
parsed.declared_nonzero_count # 2
parsed.offset                 # Decimal('7')
[(t.i, t.j, t.coefficient) for t in parsed.terms]
# [(1, 1, Decimal('-2')), (1, 2, Decimal('3'))]
```

**Assumptions and validation:**
- `1 <= i <= j <= n`; indices start at 1 and **only the stored upper half** is allowed.
- The number of data rows must match the declared count; pairs must be unique.
- Coefficients and offsets must be finite decimal numbers; a supposedly nonzero stored entry cannot be zero.
- A variable does **not** need a diagonal entry or any interaction.
- Missing offset remains `None`.
- `# minimize` or `# maximize` is recorded as a **declared direction from comments**, not automatically inferred as the stored objective direction.
- Preamble comments are retained for provenance, not interpreted as authoritative mathematical semantics.

**Deliberately NOT done here:** No factor of two, sign flip, symmetrization, energy evaluation, solver calls, or directory traversal.

**Next in chain:** `model.from_qs(parsed)`; or simply use `model.load_qs(path)` to do both stages.

**Tests:** Structural parsing and bad-index handling are exercised through `tests/test_qubo_model.py` (and its legacy inventory check). A separate `test_qubo_qs.py` was **not included** in the supplied snapshot.

### 3.2 `model.py` — canonical source-preserving record (Layer 2)

**Position:** Bridges `qs.py` to the rest of the reader. Owns raw instance identity, metadata and sparse structure.

**Public objects/functions:**
- `QuboTerm(i, j, coefficient)`: an unchanged raw stored triplet
- `QuboInstance`: canonical sparse raw record; `@dataclass(frozen=True, slots=True)`
- `from_qs(parsed, *, family="unknown", instance_id=None) -> QuboInstance`
- `load_qs(path, *, dataset_root=None, family=None, instance_id=None) -> QuboInstance`

**Input:** A `QSInstance` for `from_qs`, or a `.qs` path for `load_qs`.

**Output example:**

```python
raw = load_qs(path)
raw.num_variables      # 4
raw.num_stored_terms   # 2
raw.num_diagonal_terms # 1
raw.num_interactions   # 1
raw.raw_diagonal       # {1: Decimal('-2')}
raw.raw_off_diagonal   # {(1, 2): Decimal('3')}
list(raw.variables)    # [1, 2, 3, 4]
raw.offset             # Decimal('7')
```

**Provenance:** `instance_id`, `family`, `source_path`, `source_format`, `declared_direction`, source comments, and explicitly unresolved `stored_direction='unknown'`, `direction_relation='unknown'`, `coefficient_convention='unverified'`. Counts and `graph_density` are derived from stored sparse terms. Derived dictionaries are freshly constructed rather than shared mutable views.

**Optional path metadata:** With `dataset_root`, family and a family-relative instance ID are inferred from the relative path; the file must live under that root in an expected `root/<family>/.../<file>.qs` layout. This is **metadata inference for one file**, not bulk loading.

**Assumptions and design choices:**
- The header's declared variable count determines the problem dimension. Do **not** estimate it from the number of diagonal coefficients or vertices in an edge list.
- Stored triplets preserve their values and order, with no semantic interpretation.
- Data-class fields are frozen; the underlying representation remains sparse. This does **not** make subsequent `InterpretedQubo` or `ConvertedQubo` immutable.
- Structural checks validate the dimension, uniqueness, nonzero finite terms, and expected 1-based upper-half storage.

**Next in chain:** `interpret_qubo(raw, ...)`; the alternative **structural side path** is `graph.py`.

**Tests:** Six methods in `tests/test_qubo_model.py`, including data preservation, missing diagonals and isolated variables, metadata, error propagation, immutability, and a legacy `inventory_row` compatibility test. That last test imports `inventory.py`, which is intentionally **outside this README's implementation scope**.

### 3.3 `graph.py` — raw structural inspection (Layer 3a; optional side path)

**Position:** Takes a raw `QuboInstance` directly. It is **not** part of the mandatory route from raw QUBO to `ConvertedQubo`.

**Public functions:**

| Function | Input | Output | Coefficient meaning |
|---|---|---|---|
| `to_edge_list(instance, include_coefficients=False)` | Raw `QuboInstance` | Tuple of `(i,j)` pairs; optionally `(i,j,raw_value)` | Raw stored value when included |
| `to_interaction_dict(instance)` | Raw `QuboInstance` | Fresh `{(i,j): raw_value}` mapping | Raw stored value |
| `to_networkx(instance)` | Raw `QuboInstance` | Undirected `nx.Graph` | Raw stored values in named attributes |

**Input / output example:**

```python
from my_code.qubo_reader.graph import to_edge_list, to_interaction_dict, to_networkx

to_edge_list(raw)  # ((1, 2),)
to_interaction_dict(raw)  # {(1, 2): Decimal('3')}
G_raw = to_networkx(raw)
G_raw.number_of_nodes()  # 4
G_raw.number_of_edges()  # 1
G_raw.edges[1, 2]['raw_coefficient']  # Decimal('3')
G_raw.nodes[1]['raw_diagonal_coefficient']  # Decimal('-2')
```

**Rules and rationale:**
- Off-diagonal terms define graph edges. Diagonal terms are **node attributes**, not self-loops.
- NetworkX receives all declared nodes, even isolated ones.
- `to_edge_list` alone cannot encode isolated variables: carry `num_variables` separately if needed.
- Coefficients are labeled `raw_coefficient`, not the standard `weight`, to avoid implying a normalized QUBO coupling.
- `to_networkx` records instance ID, family, path, offset, declared/stored directions and coefficient status in `G_raw.graph`. These are provenance/assumption labels, **not** a proof of energy semantics.
- No source-convention inference, multiplying by two, sign inversion or objective evaluation.

**What about `adapters.to_networkx_graph`?** That is a **different** function: it takes `ConvertedQubo` and optionally stores **converted** coefficients. Keep the two functions distinct; do not substitute one for the other without checking the intended coefficient meaning.

**Tests:** Six methods in `tests/test_qubo_graph.py` cover raw coefficients, fresh dictionaries, isolated nodes, missing terms, graph counts, and non-inference of the declared optimization direction.

### 3.4 `conventions.py` — mathematical interpretation and conversion (Layer 3b)

**Position:** Between the raw record and output adapters. It is the only layer deliberately changing mathematical coefficient representation or optimization direction.

**Public classes/functions:**
- `interpret_qubo(raw, *, source_convention, source_direction, offset_override=None, assumption_note="") -> InterpretedQubo`
- `convert_qubo(interpreted, *, target_convention, target_direction) -> ConvertedQubo`
- `evaluate_interpreted(interpreted, assignment) -> Decimal`
- `InterpretedQubo`: explicit source assumptions, full polynomial coefficients (`linear`, `quadratic`), offset, direction, and reference to raw.
- `ConvertedQubo`: requested target matrix entries (`coefficients`), offset, convention, direction, `energy_sign`, and reference to interpreted.

**Step A — interpret the raw numbers:**

```python
interpreted = interpret_qubo(
    raw,
    source_convention='upper_triangular',
    source_direction='minimize',
    assumption_note='Explicit example assumption',
)
interpreted.linear       # {1: Decimal('-2')}
interpreted.quadratic    # {(1, 2): Decimal('3')}
interpreted.offset       # Decimal('7')
interpreted.direction    # 'minimize'
```

`source_convention='symmetric'` would produce `quadratic={(1,2): Decimal('6')}` instead. `source_direction` records the assumed **optimization sense**, but does **not** itself flip input coefficient signs.

**Step B — convert the interpreted objective to a requested representation:**

```python
converted = convert_qubo(
    interpreted,
    target_convention='symmetric',
    target_direction='maximize',
)
converted.coefficients   # {(1, 1): Decimal('2'), (1, 2): Decimal('-1.5')}
converted.offset         # Decimal('-7')
converted.energy_sign    # -1
```

The numerical relationship is `E_converted(x) = -E_interpreted(x)` for this direction flip, but `ConvertedQubo.coefficients[(1,2)] == -1.5` because the target is **symmetric**: its polynomial interaction contribution is **`2 * (-1.5) * x1 * x2 = -3 * x1 * x2`**.

**Step C — evaluate for diagnostics:**

```python
evaluate_interpreted(interpreted, [1, 1, 0, 0])  # Decimal('8')
```

This function evaluates the **interpreted** polynomial and includes the offset. It is for internal checking; `adapters.to_objective_function()` is the solver-facing callable built from a `ConvertedQubo`.

**Validation and design choices:**
- Source and target conventions must be one of `symmetric`, `upper_triangular`; directions must be `minimize`, `maximize`.
- Source assumptions are **required**, not silently copied from source comments.
- Missing source offset must be resolved explicitly with `offset_override`.
- Offset overrides exclude floats to avoid accidental decimal conversion artifacts.
- The conversion changes the **representation**, not which assignments are optimal (when direction is inverted, min and max are exchanged).
- The original `QuboInstance` and `InterpretedQubo` are not mutated by `convert_qubo`.
- `InterpretedQubo` and `ConvertedQubo` are **ordinary mutable Python classes**; treat them as snapshots by convention.

**Tests:** Seven methods in `tests/test_qubo_conventions.py`, most notably the exhaustive **16 source/target convention-and-direction combinations** over **all 16 binary assignments** for a four-variable fixture. This proves the algebraic relationship **for the tested fixtures**, conditional on the chosen source interpretation; it does not prove the original dataset author's intent.

### 3.5 `adapters.py` — serve the chosen representation (Layer 4)

**Position:** Final layer for a single converted instance. An adapter changes **data structure**, not source interpretation or target mathematical convention.

**Input shared by all four adapters:** `ConvertedQubo`. The caller first chooses a target convention and direction through `convert_qubo`.

| Adapter | Required target | Output | Core guarantees / limitations |
|---|---|---|---|
| `to_dimod_bqm(converted)` | `upper_triangular`, `minimize` | `dimod.BinaryQuadraticModel` with `BINARY` variables | All 1..n variables; diagonal → linear; interactions → quadratic; offset included; Decimal → float (precision may be lost) |
| `to_networkx_graph(converted, *, include_metadata=False)` | Any valid `ConvertedQubo` | `networkx.Graph` | All nodes and nonzero interaction edges; optional converted coefficients + provenance/offset; not a standard energy-evaluation API |
| `to_objective_function(converted)` | Any valid `ConvertedQubo` | Callable `f(assignment) -> Decimal` | Evaluates target objective including offset; understands symmetric factor 2; accepts n-bit sequence or full `{1..n: bit}` dict |
| `to_mqlib_file(converted)` | `symmetric`, `maximize` | Context manager yielding `pathlib.Path` | Writes sparse text `n m`, `i j coefficient`; offset **not** in file; path valid only within caller's `with` block |

**Output examples for the same running instance:**

```python
from my_code.qubo_reader.adapters import (
    to_dimod_bqm, to_networkx_graph,
    to_objective_function, to_mqlib_file,
)

# Upper-triangular, minimize target:
converted_min = convert_qubo(interpreted,
    target_convention='upper_triangular', target_direction='minimize')
bqm = to_dimod_bqm(converted_min)
# bqm.linear[1] == -2.0, bqm.quadratic[1, 2] == 3.0,
# bqm.offset == 7.0, variables == {1, 2, 3, 4}

G = to_networkx_graph(converted_min, include_metadata=True)
# G has 4 nodes and 1 edge.
# G.nodes[1]['linear_coefficient'] == Decimal('-2')
# G.edges[1, 2]['quadratic_coefficient'] == Decimal('3')
# G.graph['offset'] == Decimal('7')

f = to_objective_function(converted_min)
# f([1, 1, 0, 0]) == Decimal('8')

# Symmetric, maximize target:
converted_max = convert_qubo(interpreted,
    target_convention='symmetric', target_direction='maximize')
with to_mqlib_file(converted_max) as path:
    print(path.read_text(encoding='utf-8'))
    # 4 2
    # 1 1 2
    # 1 2 -1.5
# The temporary file and its containing directory are now gone.
```

**`to_networkx_graph` metadata behavior:**

- `include_metadata=False` (default): structural graph with all declared nodes, only nonzero off-diagonal edges, and **no** extra QUBO attributes.
- `include_metadata=True`: node `linear_coefficient` (when present), edge `quadratic_coefficient`, and graph-level `instance_id`, `family`, `source_path`, `num_variables`, `index_base`, `convention`, `direction`, `offset`, `energy_sign`.
- `quadratic_coefficient` is a **matrix entry in the selected target convention**, not automatically a complete polynomial interaction coefficient. It may need the factor of 2 when the stored target convention is symmetric.
- NetworkX can store arbitrary attributes, but consumers such as graph algorithms need not interpret them as an objective. This is a **semantics-of-consumption** limitation, not an inherent NetworkX storage limitation.

**`to_objective_function` details:**

- Each stored `(i,i)` contributes `coefficient * x_i² = coefficient * x_i`; no special linear loop is needed.
- Each stored `(i,j)`, `i<j`, contributes `coefficient * x_i * x_j` in upper-triangular convention, or **twice** that in symmetric convention.
- Returns `Decimal` energy with the target offset.
- Valid assignments are **exactly** n native-Python `int` bits (0 or 1), in order, or a dictionary keyed 1..n. In the present implementation, `bool` values are rejected.

**`to_mqlib_file` details and lifecycle:**

- `@contextmanager` makes `with to_mqlib_file(converted) as path:` the required use form.
- Before `yield path`, the implementation validates the target, sorts nonzero coefficient entries, creates a unique `TemporaryDirectory`, writes and closes a `.qubo` file, then yields its path.
- While the caller's `with` body executes, the generator is **suspended** and the file remains on disk. Exiting the block resumes cleanup, including when the body raises an ordinary exception.
- Example danger: creating a file in a `with`, calling `pool.apply_async(...)`, and **leaving the `with` before the asynchronous job finishes** could delete the file too early.
- The Q_Opt design discussed separately creates per-worker temporary files during its synchronous MQLib subprocess call; a worker-local `with` could fit that pattern. This README does **not** claim the current reader is already integrated with Q_Opt.
- Cleanup on normal exit or Python exception is tested. Abrupt worker termination (`SIGKILL`), power failure, or external file deletion are outside that guarantee.

**MQLib-specific caveats:** The adapter currently **assumes** a symmetric-matrix, maximization QUBO input contract and writes only one record per pair (`i <= j`). This is a chosen and tested **serialization contract**, **not yet a completed executable-level semantic validation**. The input file has no objective offset field; keep `converted.offset` alongside it and restore it consistently when interpreting solver results. An existing Q_Opt `MQLibInterface` may also apply its own sign/scaling patch; avoid an accidental **second** conversion during future integration.

**Tests:** Thirteen methods in `tests/test_qubo_adapter.py` (three BQM, three NetworkX, three objective function, four MQLib temporary file); details below.

## 4. Worked end-to-end numerical trace

The following sequence uses the running toy file. It makes every mathematical change explicit.

| Stage | `offset` | `(1,1)` | `(1,2)` | Meaning |
|---|---:|---:|---:|---|
| Raw stored `.qs` | `7` | `-2` | `3` | No semantic interpretation yet |
| Interpreted as `upper_triangular`, `minimize` | `7` | linear `-2` | polynomial `+3` | `E=7-2x1+3x1x2` |
| Converted to `upper_triangular`, `minimize` | `7` | `-2` | `3` | Same energy, BQM-ready target |
| Converted to `symmetric`, `maximize` | `-7` | `+2` | `-1.5` | `E'=-7+2x1-3x1x2 = -E` |

For `x=(1,1,0,0)`:

- Interpreted/BQM-target energy: `7 - 2 + 3 = 8`.
- Symmetric/maximize target energy: `-7 + 2 + 2*(-1.5) = -8`.
- The MQLib file carries only the **non-offset** terms `+2` and `-1.5`; **if** the executable evaluates those according to the assumed symmetric matrix convention, its corresponding raw objective for this assignment would be `-1`, and adding `converted.offset = -7` gives `-8`. This **conditional statement** must be confirmed against MQLib itself in Phase B.

The `4 2` file header preserves the **four declared variables** even though only two nonzero matrix entries are written. Variables 3 and 4 remain part of the declared dimension.

## 5. Tests: precise coverage and evidence

The supplied snapshot contains four test files with **32 test methods in total** (6 + 6 + 7 + 13). Most create small `.qs` fixtures in temporary directories; they do not require the full QOBLib dataset.

### `tests/test_qubo_model.py` — 6 methods

| Test | What it establishes |
|---|---|
| `test_labs_raw_and_reuse` | Raw coefficients, metadata/comments, offset, counts and frozen record behavior |
| `test_mis_does_not_infer_sense_or_double` | Source `maximize` comment is not treated as encoded direction; no automatic doubling |
| `test_missing_diagonals_do_not_reduce_dimension` | Dimension survives absent diagonal entries |
| `test_absent_variable_is_still_declared` | Isolated variables persist; missing offset stays `None` |
| `test_legacy_inventory_still_works` | Existing inventory row construction remains compatible with the raw model |
| `test_error_cases_inherited_from_qs` | Bad indices and inconsistent family/path metadata are rejected |

### `tests/test_qubo_graph.py` — 6 methods

| Test | What it establishes |
|---|---|
| `test_sparse_raw_edges_and_no_factor_two` | Edges and raw stored coefficients are preserved |
| `test_interaction_dictionary_does_not_modify_instance` | Returned dictionary is independent of the source object |
| `test_networkx_includes_all_isolated_nodes` | All declared nodes, raw edge and diagonal attributes, no misleading `weight` |
| `test_mis_source_maximize_is_not_inferred_as_energy_maximize` | Graph metadata distinguishes declared and unknown stored direction |
| `test_dimension_with_no_terms` | Graph with nodes but no edges is supported |
| `test_graph_counts_match_canonical_metrics` | Graph node/edge counts agree with raw model |

### `tests/test_qubo_conventions.py` — 7 methods

| Test | What it establishes |
|---|---|
| `test_two_interpretations_of_exactly_the_same_raw_input` | Different assumptions give different polynomial energies without changing raw data |
| `test_all_sixteen_source_target_combinations` | All 2×2×2×2 convention/direction combinations, all toy assignments, preserve `E_target = sign * E_source` |
| `test_specific_symmetric_min_to_triangular_max` | Exact factor-two, sign, and offset transformation in a hand-checkable example |
| `test_comment_is_not_assumed_to_be_stored_direction` | A source declaration does not override an explicit interpretation |
| `test_offset_must_be_explicit_when_source_is_missing` | Missing offset requires an explicit choice |
| `test_invalid_inputs_rejected` | Unsupported conventions/directions, float offsets and bad assignments are rejected |
| `test_absent_diagonal_and_isolated_variable_keep_dimension` | Dimensions survive sparse/missing diagonal entries |

### `tests/test_qubo_adapter.py` — 13 methods

| Test | What it establishes |
|---|---|
| `test_creates_bqm_and_preserves_variables` | BQM biases/offset, isolated nodes, exact toy energy enumeration |
| `test_symmetric_source_is_not_double_counted` | Normalized interaction is not multiplied twice by the BQM adapter |
| `test_rejects_incompatible_conventions` | BQM rejects unsupported matrix/direction combinations |
| `test_networkx_graph_without_metadata` | Structure-only output and optional metadata default |
| `test_networkx_graph_with_metadata` | Correct converted node/edge coefficients and graph-level metadata |
| `test_networkx_does_not_change_conventions` | NetworkX adapter does not flip signs or rescale coefficients |
| `test_objective_function_energy_equivalence` | Objective callback equals reference interpreted energy after requested sign change |
| `test_objective_function_accepts_dictionary` | Sequence and dictionary assignments produce the same target energy |
| `test_objective_function_rejects_invalid_assignments` | Wrong dimensions, bad bits, incomplete keys, booleans rejected |
| `test_mqlib_file_content` | Header, sparse entries, target signs/scaling and off-file offset |
| `test_mqlib_file_exists_only_inside_with` | Temp file/directory exist only during context lifetime |
| `test_mqlib_file_cleanup_when_caller_raises` | Cleanup when caller raises an ordinary exception |
| `test_mqlib_file_rejects_incompatible_input` | MQLib adapter rejects wrong convention/direction/type |

### Test commands and status

From the QOBLIB repository root:

```bash
python3 -m unittest discover -s tests -p 'test_qubo_adapter.py' -v
python3 -m unittest discover -s tests -p 'test_qubo_*.py' -v
```

- **User-confirmed:** `test_qubo_adapter.py` ran **13 tests, `OK`** (2026-11-16).
- **Additional audit using the supplied source text:** `test_qubo_conventions.py` (**7 OK**) and `test_qubo_graph.py` (**6 OK**) were reproduced in an isolated Python test reconstruction (2026-11-16).
- **Not independently rerun in that audit:** The six `test_qubo_model.py` methods; one imports `my_code.qubo_reader.inventory.inventory_row`, whose current source was not supplied for this documentation task.
- **Not yet proven by these tests:** Real QOBLib family-wide semantic correctness, MQLib executable input semantics/output energies, reliable downstream Q_Opt integration, or concurrent real-solver execution.

**Environment:** The code needs Python **3.10+** (e.g., `dataclass(slots=True)` and modern typing), plus `networkx` and `dimod` for the adapters/test suite. The standard-library modules `decimal`, `dataclasses`, `pathlib`, `tempfile`, `contextlib`, and `unittest` require no separate pip installation. An environment lockfile or pinned dependency versions should be created in Phase A's Git/dependency checkpoint, based on the actual environment; **do not assume unverified package versions**.

## 6. Design decisions we intentionally keep

1. **No semantic guessing.** Source matrix convention and encoded optimization direction are explicit caller decisions. Source comments are informative but not authoritative for encoded energy.
2. **Raw data is preserved.** `QuboInstance` stores 1-based sparse records as seen on disk; interpretation and conversion are separate objects.
3. **No silent offset default.** Missing, zero, and overridden offsets are distinct states.
4. **Exact decimal arithmetic until an interface requires otherwise.** The `dimod` adapter deliberately crosses to `float`, so its numerical precision is not identical to `Decimal` in general.
5. **Separation of mathematical and structural concerns.** Raw structural graphs do not claim to be energy-equivalent QUBO objects; converted NetworkX graphs may carry numerical attributes, but consumers must opt to interpret them.
6. **Adapters are organized by output data type, not by solver name.** One `dimod.BQM` adapter can serve multiple consumers; new consumers need a new adapter only if the required representation differs.
7. **Adapters do not perform reductions.** In particular, this phase does **not** convert general QUBO to Max-Cut.
8. **No hidden convention changes in adapters.** Each adapter either accepts compatible `ConvertedQubo` or rejects it. The objective callback evaluates the chosen convention but does not reinterpret the source.
9. **File lifecycle is explicit.** The temporary MQLib file is available inside a context and automatically cleaned up on normal exit/ordinary exceptions. The caller owns the timing of `with` relative to subprocess completion.
10. **Incremental, readable development.** Add and understand one adapter, then its tests, rather than generating a large framework or solver registry prematurely.

## 7. Limitations, boundaries, and open verification tasks

**Known by design:**
- One `.qs` input at a time. No recursive directory reader or family-wide assumption manager yet.
- `graph.py` and `adapters.py` both expose NetworkX graph functions with different input types and coefficient meanings. This is intentional at present; keep names and provenance explicit.
- `ConvertedQubo` is not a dense matrix; the symmetric convention is **implied** by its upper-half stored pairs.
- `to_networkx_graph` does not produce a standard objective-evaluating graph object; attributes must be understood by the consumer.
- `to_mqlib_file` writes no source comments or offset; it yields a temporary path, not a persistent output artifact.
- `to_objective_function` accepts only binary assignments, not continuous vectors from a general-purpose optimizer. Discretization or solver-specific variable encoding would be a separate concern.

**Must be validated before scientific use:**
- MQLib's exact interpretation of QUBO sparse off-diagonal entries, optimization direction, and returned objective values on an **executable-level reference example**. Do **not** infer that a successful file-write test proves these semantics.
- Correct restoration of the **target offset** and, where needed, the interpreted/source energy direction when comparing solver outputs.
- Actual execution of one real instance from `.qs` through BQM, objective callback and MQLib file, against independently computed objective values.
- Numerical behavior when large/precise `Decimal` coefficients are passed through float-based consumers.
- Integration with the older Q_Opt MQLib wrapper, which currently may apply its **own** source sign/scale adjustment. It must not adjust already-converted data again.

**Not part of the present milestone:**
- Bulk/directory loading, Q_Opt pipeline refactor, multiprocessing/job scheduling, benchmark time budgets, solution decoding back to original MIPs, quantum hardware runs, and QUBO↔Max-Cut reductions.

## 8. Complete single-instance usage example

This example is **self-contained**, starts with a real `.qs` text file, follows the complete reader chain, and produces the different output data structures. It intentionally does **not** invoke a solver. Run it from the QOBLIB repository root after installing `dimod` and `networkx`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from my_code.qubo_reader.qs import parse_qs
from my_code.qubo_reader.model import from_qs
from my_code.qubo_reader.graph import to_networkx as raw_to_networkx
from my_code.qubo_reader.conventions import (
    interpret_qubo, convert_qubo, evaluate_interpreted,
)
from my_code.qubo_reader.adapters import (
    to_dimod_bqm, to_networkx_graph,
    to_objective_function, to_mqlib_file,
)

with TemporaryDirectory() as directory:
    # 1. Source file
    path = Path(directory) / "toy.qs"
    path.write_text(
        "# ObjectiveOffset 7\n"
        "# Vars Non-zeros\n"
        "4 2\n"
        "1 1 -2\n"
        "1 2 3\n",
        encoding="utf-8",
    )

    # 2. Parse -> model
    parsed = parse_qs(path)
    raw = from_qs(parsed, family="toy")
    assert raw.num_variables == 4
    assert raw.num_interactions == 1

    # Optional structural side path from RAW data
    raw_graph = raw_to_networkx(raw)
    assert raw_graph.number_of_nodes() == 4
    assert raw_graph.edges[1, 2]["raw_coefficient"] == 3

    # 3. Source interpretation (a stated assumption, not inferred)
    interpreted = interpret_qubo(
        raw,
        source_convention="upper_triangular",
        source_direction="minimize",
        assumption_note="Toy file: assume polynomial upper triangle",
    )
    x = [1, 1, 0, 0]
    assert evaluate_interpreted(interpreted, x) == 8

    # 4A. Convert for dimod / BQM-style minimization
    for_bqm = convert_qubo(
        interpreted,
        target_convention="upper_triangular",
        target_direction="minimize",
    )
    bqm = to_dimod_bqm(for_bqm)
    assert bqm.energy({1: 1, 2: 1, 3: 0, 4: 0}) == 8

    # 4B. Convert for MQLib-style symmetric maximization
    for_mqlib = convert_qubo(
        interpreted,
        target_convention="symmetric",
        target_direction="maximize",
    )
    f = to_objective_function(for_mqlib)
    assert f(x) == -8

    graph = to_networkx_graph(for_mqlib, include_metadata=True)
    assert graph.number_of_nodes() == 4
    assert graph.edges[1, 2]["quadratic_coefficient"] == -1.5
    assert graph.graph["offset"] == -7

    with to_mqlib_file(for_mqlib) as mqlib_path:
        assert mqlib_path.is_file()
        print(mqlib_path.read_text(encoding="utf-8"))
        # 4 2
        # 1 1 2
        # 1 2 -1.5
        # Any future synchronous MQLib use of this path must finish HERE.

    assert not mqlib_path.exists()

# Temporary source toy file is also gone.
```

For the existing extracted real dataset, the first two steps can be shortened to:

```python
from my_code.qubo_reader.model import load_qs
raw = load_qs(
    "extracted_qb/labs/labs006/labs006.qs",
    dataset_root="extracted_qb",
)
# Previous dataset observations: 21 variables, 117 interactions, offset 55.
# Choose source assumptions only with documented justification.
```

## 9. Current checkpoint and roadmap

### Phase A — Knowledge / development lock (current)

- [x] Consolidate the **design** into one authoritative README for these five modules (this file).
- [ ] Review this README against the installed source files and test names; correct any source-version mismatch.
- [ ] Run the **complete** reader test suite in the intended Python environment; capture its output.
- [ ] Remove older fragmented READMEs **only after** reviewing them against this replacement.
- [ ] Inspect Git status, branch and remotes; update `.gitignore`; add a minimal reproducible dependency specification; commit and push to the intended repository/branch.

### Phase B — Real-instance end-to-end validation (next; exit after completion)

- [ ] Select real QOBLib inputs, initially `labs006`, with explicit **source interpretation evidence**.
- [ ] Verify raw `.qs` numbers, declared dimensions, offset, provenance, parsed model and structure.
- [ ] Verify `InterpretedQubo` and every relevant target convention/direction mapping.
- [ ] Check numeric energy relations, on controlled assignments and via independent reference calculations; include offset and direction reversal.
- [ ] Validate all output structures (`dimod.BQM`, raw/converted NetworkX views, objective callback, serialized MQLib file).
- [ ] Run a minimal **real MQLib executable** test to confirm the file convention and returned energy/offset handling.
- [ ] Record any corrections and make a second Git checkpoint if Phase B changes code or conclusions.

**Exit criterion:** The single-instance processing chain is demonstrably correct *under explicit, justified source assumptions*, with the external file/solver semantics tested where they matter, and with a reproducible test record. Do not call semantic uncertainty “validated” merely because structural parsing succeeds.

### Phase C — Bulk/directory reader (deferred)

After A + B: accept a root folder and **one explicit fixed source interpretation** for a collection, recursively load the instances consistently, and expose results for the already-built adapters. This work is not part of the current active window.

---

**Maintenance rule:** This README documents the reader's **current code**. When functions, assumptions or tests change, update their corresponding sections here. Keep historical decisions only where they still explain current behavior; do not leave obsolete “not implemented yet” statements from v0.2–v0.4 as if they were current.
