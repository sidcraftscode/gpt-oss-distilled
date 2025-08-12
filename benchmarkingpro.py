#!/usr/bin/env python3
import argparse, json, os, shutil, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ---------------------- Config ----------------------
DEFAULT_DTYPE = "bfloat16"
OUTDIR = Path("./lighteval_live")
PORT = 8008

# Decode caps tuned for speed on A100 80GB
TOKENS = {
    "gsm8k": 1024,
    "math": 1024,
    "humaneval": 1024,
    "mt_bench": 1024,
    "openrewrite_eval": 1024,
    "nq_open_heldout": 1024,
    "triviaqa_heldout": 1024,
    "_default_mc": 1024,
}

# Batch sizes for A100 80GB (adjust down if needed)
BATCH = {
    "mc": 64,          # HellaSwag/ARC/PIQA/etc.
    "mmlu": 64,
    "bbh": 32,
    "gen_small": 16,   # short generations (NQ/Trivia/Rewrite/MT-Bench)
    "gsm8k": 8,
    "math": 6,
    "humaneval": 8,
}

# Full SmolLM2 suite (two tables merged). (task_id, shots, kind)
TASKS = [
    # Base / QA / commonsense
    ("hellaswag", 0, "mc"),
    ("arc", 0, "mc"),                # (LightEval usually handles easy+challenge)
    ("piqa", 0, "mc"),
    ("commonsense_qa", 0, "mc"),
    ("winogrande", 0, "mc"),
    ("openbook_qa", 0, "mc"),
    ("mmlu_pro", 0, "mmlu"),
    ("nq_open_heldout", 0, "gen_small"),   # fallback later if not found
    ("triviaqa_heldout", 0, "gen_small"),  # fallback later if not found
    ("gsm8k", 5, "gsm8k"),
    ("math", 4, "math"),
    ("humaneval", 0, "humaneval"),
    # Instruction / chat-type
    ("ifeval", 0, "gen_small"),
    ("mt_bench", 0, "gen_small"),
    ("openrewrite_eval", 0, "gen_small"),
    ("bbh", 3, "bbh"),
    # (mmlu_pro, hellaswag, piqa, gsm8k, math, humaneval already covered)
]

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>LightEval – Live Benchmarks</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
  h1 { margin: 0 0 8px; }
  .sub { color:#555; margin:0 0 16px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border-bottom: 1px solid #eee; padding: 10px; text-align: left; }
  th { background: #fafafa; position: sticky; top: 0; }
  tr.done td { background: #fcfffc; }
  .ok { color: #208120; font-weight: 600; }
  .warn { color: #a66d00; }
  .err { color: #b00020; font-weight: 600; }
  .mono { font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  footer { color:#777; margin-top:16px; font-size: 12px; }
</style>
</head>
<body>
  <h1>LightEval – Live Benchmarks</h1>
  <p class="sub">Auto-refreshing every 3s. Results file: <code>results.json</code>.</p>
  <table id="tbl">
    <thead>
      <tr>
        <th>#</th><th>Task</th><th>Shots</th><th>Score</th><th>Status</th><th>Runtime</th><th>Report</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
  <footer id="meta"></footer>
<script>
async function load() {
  try {
    const res = await fetch('results.json?ts=' + Date.now());
    const data = await res.json();
    const tb = document.querySelector('#tbl tbody');
    tb.innerHTML = '';
    data.tasks.forEach((t, i) => {
      const tr = document.createElement('tr');
      if (t.status === 'done') tr.classList.add('done');
      const scoreDisp = (t.score === null || t.score === undefined) ? '—' : t.score.toFixed(3);
      const link = t.report && t.report.exists ? `<a href="${t.report.path}" target="_blank">JSON</a>` : '—';
      tr.innerHTML = `
        <td class="mono">${i+1}</td>
        <td>${t.name}</td>
        <td class="mono">${t.shots}</td>
        <td class="mono">${scoreDisp}</td>
        <td>${t.status === 'done' ? '<span class="ok">done</span>' :
               t.status === 'running' ? '<span class="warn">running…</span>' :
               t.status === 'skipped' ? 'skipped' : '<span class="err">error</span>'}</td>
        <td class="mono">${t.runtime || '—'}</td>
        <td>${link}</td>`;
      tb.appendChild(tr);
    });
    document.querySelector('#meta').textContent =
      `Model: ${data.model} | dtype: ${data.dtype} | batch: ${data.batch_hint} | Completed: ${data.done}/${data.total} | Elapsed: ${data.elapsed_human}`;
  } catch(e) {
    console.error(e);
  }
}
load();
setInterval(load, 3000);
</script>
</body>
</html>
"""

# ---------------------- Helpers ----------------------
def humantime(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def ensure_installed():
    try:
        import lighteval  # noqa
    except Exception:
        print("[setup] Installing LightEval and deps…", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lighteval[all]", "transformers", "accelerate",
                               "datasets", "evaluate", "sentencepiece", "huggingface_hub"])

def write_index(outdir: Path):
    (outdir / "index.html").write_text(HTML_TEMPLATE, encoding="utf-8")

def start_server(root: Path, port: int):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd

def write_results_json(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def task_display_name(tid):
    return tid

def score_from_result(obj: dict):
    # LightEval stores metrics per task; try common keys
    for k in ("accuracy", "acc", "rougeL", "pass@1", "exact_match", "score"):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # sometimes nested under "metrics"
    met = obj.get("metrics") or {}
    for k in ("accuracy", "acc", "rougeL", "pass@1", "em", "score"):
        if k in met:
            return float(met[k])
    return None

def ensure_custom_openrewrite(task_dir: Path):
    p = task_dir / "openrewrite_task.py"
    if p.exists():
        return
    task_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(r'''
from typing import List, Dict, Any
from datasets import load_dataset
from lighteval.tasks import Task, Instance, register_task
from lighteval.metrics.rouge import RougeL

@register_task("openrewrite_eval")
class OpenRewriteEval(Task):
    def __init__(self, split: str = "test", **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.rouge = RougeL()
    def load(self):
        try:
            ds = load_dataset("argilla/OpenRewrite-Eval", split=self.split)
        except Exception:
            ds = load_dataset("argilla/openrewrite_eval", split=self.split)
        self.data = []
        for i, ex in enumerate(ds):
            prompt = ("Rewrite the text to be clearer, friendlier, and concise, "
                      "preserving meaning. Return only the rewritten text.\n\nText:\n" + ex["source"])
            self.data.append(Instance(input=prompt, references=[ex["reference"]], id=str(ex.get("id", i))))
    def instances(self) -> List[Instance]:
        return self.data
    def process_results(self, instance: Instance, model_output: str) -> Dict[str, Any]:
        score = self.rouge([model_output], [instance.references[0]])
        return {"rougeL": score}
    def aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        vals = [r["rougeL"] for r in results]
        return {"rougeL": sum(vals)/max(1,len(vals))}
''', encoding="utf-8")

# ---------------------- Runner ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./results/cece-1.7b", help="Path to local HF model folder")
    ap.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["float16","bfloat16","float32"])
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--tensor-parallel", type=int, default=1)
    args = ap.parse_args()

    ensure_installed()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # write dashboard + placeholder results
    write_index(out)
    res_path = out / "results.json"
    start = time.time()

    # custom OpenRewrite task
    custom_dir = out / "custom_tasks"
    ensure_custom_openrewrite(custom_dir)

    # start server
    srv = start_server(out, args.port)
    print(f"[server] Live dashboard at http://localhost:{args.port}/", flush=True)

    # function to dump status
    tasks_state = []
    for tid, shots, kind in TASKS:
        tasks_state.append({
            "name": task_display_name(tid),
            "id": tid, "shots": shots, "status": "pending",
            "score": None, "runtime": None,
            "report": {"path": f"{tid}.json", "exists": False},
        })

    def dump():
        done = sum(1 for t in tasks_state if t["status"] == "done")
        payload = {
            "model": os.path.basename(os.path.abspath(args.model)),
            "dtype": args.dtype,
            "batch_hint": BATCH,
            "total": len(tasks_state),
            "done": done,
            "elapsed_sec": int(time.time()-start),
            "elapsed_human": humantime(time.time()-start),
            "tasks": tasks_state,
        }
        write_results_json(res_path, payload)

    dump()

    # discover available task keys (to allow fallback for held-out names)
    try:
        list_out = subprocess.check_output(["lighteval", "tasks", "list"], text=True)
    except Exception:
        list_out = ""
    def available(key):
        return key in list_out

    # run tasks one-by-one to update live
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{custom_dir}{os.pathsep}{env.get('PYTHONPATH','')}"

    for idx, (tid, shots, kind) in enumerate(TASKS):
        state = tasks_state[idx]
        # fallback task names if needed
        if tid == "nq_open_heldout" and not available("nq_open_heldout"): tid = "nq_open"
        if tid == "triviaqa_heldout" and not available("triviaqa_heldout"): tid = "triviaqa"

        # per-task params
        if kind in ("mc","mmlu"): bs = BATCH["mc"]
        elif kind == "bbh": bs = BATCH["bbh"]
        elif kind == "gsm8k": bs = BATCH["gsm8k"]
        elif kind == "math": bs = BATCH["math"]
        elif kind == "humaneval": bs = BATCH["humaneval"]
        else: bs = BATCH["gen_small"]

        # tokens cap
        mtoks = TOKENS.get(tid, TOKENS["_default_mc"])
        report = out / f"{tid}.json"

        # Build model args as comma-separated string (using correct parameter names)
        # Note: max_new_tokens removed as it's not a valid model parameter in current lighteval
        model_args = f"model_name={args.model},dtype={args.dtype},batch_size={bs}"
        if args.tensor_parallel > 1:
            model_args += f",tensor_parallel={args.tensor_parallel}"
        
        # Task string with shots
        task_string = f"{tid}|{shots}"
        
        cmd = [
            "lighteval", "accelerate",
            "--output-dir", str(out),
            "--custom-tasks", str(custom_dir),
            model_args,
            task_string,
        ]

        print(f"[run] {tid} (shots={shots})  bs={bs}  max_new_tokens={mtoks}", flush=True)
        state["status"] = "running"; dump()
        t0 = time.time()
        try:
            subprocess.run(cmd, env=env, check=True)
            state["status"] = "done"
        except subprocess.CalledProcessError as e:
            state["status"] = "error"
            print(f"[error] Task {tid} failed: {e}", flush=True)

        # parse report for score
        if report.exists():
            try:
                obj = json.loads(report.read_text())
                # LightEval usually: {"results": {"<task>": {...metrics...}}}
                res = obj.get("results", {})
                # pick the first result dict
                rdict = next(iter(res.values())) if res else {}
                state["score"] = score_from_result(rdict)
                state["report"]["exists"] = True
            except Exception:
                pass

        state["runtime"] = humantime(time.time() - t0)
        dump()

    print(f"[done] All tasks finished in {humantime(time.time()-start)}")
    print(f"[open] http://localhost:{args.port}/  (press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
