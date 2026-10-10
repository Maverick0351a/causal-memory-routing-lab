"""Keyless QCA CLI wrapper with a Windows-safe subprocess adapter.

Vendor code is unmodified. Use explicit subcommands, never the conversational CLI.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "Quantum-Calibration-Agent-Blueprint"


def prepare():
    for key in list(os.environ):
        if any(s in key.upper() for s in ("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN")):
            os.environ.pop(key, None)
    os.environ.update(ARCHLAB_OFFLINE="1", LANGCHAIN_TRACING_V2="false",
                      LANGSMITH_TRACING="false", PYTHONUTF8="1")
    os.environ["PYTHONPATH"] = str(ROOT / "scripts" / "offline_guard")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    os.environ["QCAL_SCRIPTS_DIR"] = str(VENDOR / "scripts")
    os.environ.setdefault("QCAL_DATA_DIR", str(ROOT / "results" / "qca"))
    sys.path.insert(0, str(ROOT / "scripts" / "offline_guard"))
    import sitecustomize
    if os.environ["ARCHLAB_OFFLINE"] != "1":
        raise RuntimeError("offline guard missing")
    sys.path.insert(0, str(VENDOR))


def run_safe(name, params, scripts_dir, timeout=300, python_path=None, log_file=None):
    from core import discovery, runner
    schema = discovery.get_experiment_schema(name, scripts_dir)
    if schema is None:
        raise ValueError(f"Unknown experiment: {name}")
    params = runner.resolve_params(params, schema)
    errors = runner.validate_params(params, schema)
    if errors:
        raise ValueError("; ".join(errors))
    module = f"{Path(scripts_dir).name}.{Path(schema.module_path).stem}"
    seed = int(os.environ.get("QCA_SYNTHETIC_SEED", "20261010"))
    # repr() preserves backslashes and quotes in Windows paths.
    code = (
        "import importlib,json,sys,numpy as np\n"
        f"sys.path.insert(0, {str(Path(scripts_dir).parent)!r})\n"
        f"np.random.seed({seed})\n"
        f"fn=getattr(importlib.import_module({module!r}), {name!r})\n"
        "print(json.dumps(fn(**json.loads(sys.stdin.read()))))\n"
    )
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf-8")
    try:
        out, err = p.communicate(json.dumps(params), timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            p.kill()
        p.communicate()
        raise TimeoutError("Synthetic experiment timed out")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).write_text(err, encoding="utf-8")
    else:
        sys.stderr.write(err)
    if p.returncode:
        raise RuntimeError(err)
    return runner._parse_and_validate_result(out)


def main():
    allowed = {("experiments", "list"), ("experiments", "schema"),
               ("experiments", "run"), ("history", "list"), ("history", "show"),
               ("data", "arrays"), ("data", "get")}
    if tuple(sys.argv[1:3]) not in allowed:
        raise SystemExit("Use experiments list/schema/run, history list/show, data arrays/get")
    prepare()
    from core import runner
    runner.run_experiment = run_safe
    import cli
    cli.app()


if __name__ == "__main__":
    main()
