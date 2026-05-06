# SCADA Scaffolding Pipeline

Turns a `scada.md` API reference into verified `(task_description, Python_logic)` pairs,
ready to feed into the cleaning pipeline for test case generation.

---

## Quick Start

```bash
# activate env
conda activate rl

# fill in your LLM endpoint
nano scaffolding/config.json

# run on your scada.md
python -m scaffolding.main --input scada.md

# review results
python -m scaffolding.review
```

---

## Config

`scaffolding/config.json`:
```json
{
    "model": "gemma4",
    "base_url": "http://127.0.0.1:8080/v1",
    "api_key": "local",
    "max_tokens": 1024
}
```

Any OpenAI-compatible endpoint works (vLLM, llama.cpp server, OpenAI, etc.).

---

## Input: `scada.md` Format

One bullet per function. Three fields, all on one line:

```
- function_name — Short description of what it does. (Usage: How it's called.; Interacts with: subsystem1, subsystem2)
```

Example:
```
- RingBufferInsert — Inserts a data element into a circular ring buffer on the subordinate processor. (Usage: Called with the element value.; Interacts with: mpf_mfs_enq, subordinate buffer state)
```

**Do NOT put task logic in the description.** The LLM invents task variations from the raw description.
The description should describe what the real C function does — not what Python should do with files 2001/2002.

---

## Output Structure

```
output/
  {function_name}/
    task_0/
      task.json       ← task description the LLM invented
      process1.py     ← LLM-generated Python logic
      process2.py     ← only for two-process tasks
      harness.py      ← templated (not LLM-generated)
      run_log.txt     ← stdout/stderr from sandbox run
    task_1/
      ...
```

### task.json
```json
{
  "function_name": "RingBufferInsert",
  "task_id": 0,
  "description": "Read integer from file 2001, add 1, write back. File 2002 unchanged.",
  "pattern": "arithmetic",
  "primary_type": "i"
}
```

### run_log.txt
```
=== stdout ===
43
7

=== stderr ===

=== exit: 0 ===
```
Exit 0 = PASS. Anything else = FAIL.

---

## CLI Reference

### `python -m scaffolding.main`

```
--input      Path to scada.md              (default: scada.md)
--output     Output directory              (default: output/)
--config     LLM config JSON               (default: scaffolding/config.json)
--function   Process only one function by name
--n-tasks    Task variations per function  (default: 5)
--concurrency  Max concurrent workers      (default: 10)
--skip-existing  Skip functions already fully generated
```

Examples:
```bash
# all functions, 5 tasks each, 20 workers
python -m scaffolding.main --input scada.md --n-tasks 5 --concurrency 20

# one function only, 3 tasks
python -m scaffolding.main --input scada.md --function RingBufferInsert --n-tasks 3

# resume interrupted run
python -m scaffolding.main --input scada.md --n-tasks 5 --skip-existing
```

### `python -m scaffolding.review`

```
--output     Output directory              (default: output/)
--function   Show all tasks for one function
--task N     Show all files for task N (requires --function)
```

Examples:
```bash
# summary table across all functions
python -m scaffolding.review

# all tasks for one function
python -m scaffolding.review --function RingBufferInsert

# full file dump for one task
python -m scaffolding.review --function RingBufferInsert --task 2
```

---

## IO Contract (what the LLM must follow)

The harness emulates two SCADA sensor files:

| | File 2001 | File 2002 |
|---|---|---|
| Type | `int` (signed 32-bit) or `str` (≤4 chars) | same |
| Input | written by harness before process runs | same |
| Output | read by harness after process returns | same |

Rules enforced at runtime:
- **Type preserved**: int in → int out. str in → str out (≤4 chars).
- **Int range**: -2147483648 to 2147483647.
- **I ≠ O**: at least one file must change (otherwise indistinguishable from silent failure).
- **Only IO**: no file access, no network, no globals — only `state["2001"]` and `state["2002"]`.

For two-process tasks, `state["_event"]` is an `asyncio.Event`:
- Process 1 calls `await state["_event"].wait()` before reading.
- Process 2 writes data then calls `state["_event"].set()`.

---

## Layer Architecture

```
scada.md
    │
    ▼
parse_md.py          Layer 1  — extract (name, description, interacts_with)
    │
    ▼
generate_tasks.py    Layer 0.5 — LLM designs N task specs per function, decides
    │  one-process vs two-process per task, records reasoning in task.json
    │  (all functions run concurrently)
    ▼
generate_code.py     Layer 2  — reads two_process from task.json (no classification
    │  here), generates process1.py and process2.py via LLM
    │  (semaphore-bounded, p1+p2 generated in parallel for two-process tasks)
    ▼
generate_harness.py  Layer 3  — template-fills harness.py (no LLM)
    │
    ▼
run_sandbox.py       Layer 4  — subprocess with 10s timeout, writes run_log.txt
```

---

## Starting the vLLM Server (local)

```bash
conda activate rl
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --served-model-name gemma4 \
  --port 8080 \
  --dtype bfloat16 \
  --max-model-len 4096
```

Then set `config.json` → `"base_url": "http://127.0.0.1:8080/v1"`.

---

## Not Built Yet (Next Layers)

### Test Case Generator
Once a task folder is verified (PASS), call `harness.run(type1, val1, type2, val2)`
programmatically across a sweep of inputs:
- Normal values, edge cases: `0`, `-1`, `2147483647`, `-2147483648`, `""`, `"a"`, `"abcd"`
- Record `{input, expected_output}` for each call
- Drop any where `token1 == val1` AND `token2 == val2` (I == O — process did nothing observable)
- Output: list of verified test cases attached to the task folder

### SCADA Description Assembler
Takes an approved task folder + the real function entry from `scada.md`:
- Rewrites the Python logic description using real SCADA API names (from `interacts_with`)
- Produces the final JSONL row in the schema expected by the cleaning pipeline:
```json
{
  "source_function": "RingBufferInsert",
  "tasks": [
    {
      "task": "...",
      "processes": [{ "description": "...", "execution_order": 1 }],
      "test_cases": [{ "input": ["i", "42", "i", "7"], "expected_output": "43\n7" }]
    }
  ]
}
```
