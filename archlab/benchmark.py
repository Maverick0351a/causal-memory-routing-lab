"""Private benchmark generator and predict-before-feedback evaluation harness."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .public import Observation, FeedbackState, experts
from .controllers import AutoregressiveRouter, RecurrentRouter, Adam, softmax
from .memory import Memory

TRAIN_SEEDS = [101, 202, 303, 404, 505, 606]
TEST_SEEDS = [1001, 2002, 3003, 4004, 5005]
MODEL_SEEDS = [11, 29, 47]


def episode(seed, steps=768, condition="standard"):
    streams = np.random.SeedSequence(seed).spawn(4)
    x_rng, rule_rng, noise_rng, null_rng = [np.random.default_rng(s) for s in streams]
    low, high = (8, 20) if condition == "fast_switch" else (24, 64)
    rule = int(rule_rng.integers(4))
    countdown = int(rule_rng.integers(low, high+1))
    age = 0
    rows = []
    for t in range(steps):
        switched = False
        if countdown == 0:
            rule = (rule + int(rule_rng.integers(1, 4))) % 4
            countdown = int(rule_rng.integers(low, high+1))
            age = 0
            switched = True
        obs = Observation(tuple(int(x) for x in x_rng.integers(0, 2, 2)))
        label = int(experts(obs)[rule]) ^ int(noise_rng.random() < .05)
        if condition == "random_labels":
            label = int(null_rng.integers(2))
        # Only evaluator/trainer owns these tuples; controller receives public features.
        rows.append((obs, label, age, switched))
        countdown -= 1
        age += 1
    return rows


def train(model, epochs=12, steps=512):
    opt = Adam(model.params)
    rng = np.random.default_rng(700)
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for seed in rng.permutation(TRAIN_SEEDS):
            memory = FeedbackState()
            model.reset()
            caches, outputs, labels = [], [], []
            for i, (obs, y, _, _) in enumerate(episode(int(seed), steps)):
                _, cache = model.forward(memory.features(obs))
                # Forward precedes label release. Loss sees only y, never the hidden rule.
                caches.append(cache)
                outputs.append(experts(obs))
                labels.append(y)
                memory.observe(obs, y)
                if len(caches) == 16 or i == steps-1:
                    grads, loss = model.gradients(caches, outputs, labels)
                    opt.step(model.params, grads)
                    epoch_losses.append(loss)
                    caches, outputs, labels = [], [], []
        losses.append(float(np.mean(epoch_losses)))
    model.reset()
    return losses


def selected_fixed_expert():
    errors = np.zeros(4)
    for seed in TRAIN_SEEDS:
        for obs, y, _, _ in episode(seed, 512):
            errors += experts(obs) != y
    return int(np.argmin(errors)), errors.tolist()


def metrics(trace):
    p = np.array([r["probability"] for r in trace])
    y = np.array([r["label"] for r in trace])
    correct = (p >= .5) == y
    conf = np.maximum(p, 1-p)
    high = conf >= .9
    # Exclude initial episode warm-up; the protocol calls for actual switches only.
    early = np.array([r["age"] < 8 and r["step"] >= 8 for r in trace])
    ece = 0.0
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean()-correct[mask].mean())
    return {"accuracy": float(correct.mean()), "brier": float(np.mean((p-y)**2)),
            "log_loss": float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            "ece": float(ece), "high_confidence_coverage": float(high.mean()),
            "confidently_wrong_rate": float(np.mean(high & ~correct)),
            "high_confidence_error_rate": float((~correct[high]).mean()) if high.any() else None,
            "early_accuracy": float(correct[early].mean()) if early.any() else None,
            "sampled_accuracy": float(np.mean([r["sampled_correct"] for r in trace]))}


def evaluate(model, rows, *, name, fixed=0, rng_seed=0, feedback=True, persistent=None, namespace=""):
    memory = FeedbackState()
    rng = np.random.default_rng(rng_seed)
    if model is not None:
        model.reset()
    trace = []
    start = time.perf_counter()
    for step, (obs, y, age, switched) in enumerate(rows):
        x = memory.features(obs)
        if model is not None:
            q, _ = model.forward(x)
        elif name == "fixed_expert":
            q = np.eye(4)[fixed]
        elif name == "fixed_uniform":
            q = np.full(4, .25)
        elif name == "recency":
            q = softmax(-8 * memory.fast)
        else:
            raise ValueError(name)
        output = experts(obs)
        p = float(.05 + .9 * (q @ output))
        if isinstance(model, AutoregressiveRouter):
            route, log_q = model.sample(x, rng)
            action = route[0]*2+route[1]
        else:
            action = int(rng.choice(4, p=q))
            route = (action//2, action%2)
            log_q = float(np.log(q[action]))
        # Prediction is fixed above; evaluator-only values are attached below for scoring.
        trace.append({"step":step, "observation":list(obs.bits), "probability":p,
                      "label":y, "age":age, "switched":switched, "route":route,
                      "log_q":log_q, "sampled_correct":bool(output[action] == y)})
        if feedback:
            memory.observe(obs, y)
            if persistent:
                persistent.append(namespace, step, obs.bits, y)
    return metrics(trace), trace, time.perf_counter()-start


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [root/"PROTOCOL.md", *sorted((root/"archlab").glob("*.py"))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"protocol-lock.json").write_text(json.dumps(source_hashes(), indent=2))
    fixed, errors = selected_fixed_expert()
    models, training = {}, {}
    for seed in MODEL_SEEDS:
        for cls in (AutoregressiveRouter, RecurrentRouter):
            m = cls(seed)
            name = f"{m.kind}_{seed}"
            started = time.perf_counter()
            losses = train(m)
            training[name] = {"loss_by_epoch":losses, "seconds":time.perf_counter()-started,
                              "parameters":m.parameter_count}
            m.save(output/"checkpoints"/(name+".npz"))
            models[name] = m
            print(name, "trained; loss", round(losses[-1], 4), flush=True)
    result = {"protocol":1, "training_seeds":TRAIN_SEEDS, "test_seeds":TEST_SEEDS,
              "model_seeds":MODEL_SEEDS, "fixed_expert":fixed, "fixed_train_errors":errors,
              "training":training, "runs":[], "source_hashes":source_hashes()}
    store = Memory(output/"memory.sqlite")
    for condition in ("standard", "fast_switch", "random_labels", "no_feedback"):
        for seed in TEST_SEEDS:
            rows = episode(seed, 768, condition)
            all_models = {"fixed_expert":None, "fixed_uniform":None, "recency":None, **models}
            for name, model in all_models.items():
                namespace = f"{condition}/{seed}/{name}"
                persist = store if (condition == "standard" and seed == 1001 and name == "autoregressive_11") else None
                score, trace, elapsed = evaluate(model, rows, name=name, fixed=fixed,
                    rng_seed=seed+123456, feedback=condition != "no_feedback", persistent=persist, namespace=namespace)
                family = name.rsplit("_", 1)[0] if model is not None else name
                result["runs"].append({"condition":condition, "seed":seed, "model":name,
                                       "family":family, "metrics":score, "seconds":elapsed})
                trace_path = output/"traces"/condition/str(seed)/(name+".jsonl")
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text("".join(json.dumps(t)+"\n" for t in trace))
        print(condition, "evaluated", flush=True)
    store.close()
    summary = {}
    for condition in ("standard", "fast_switch", "random_labels", "no_feedback"):
        summary[condition] = {}
        for family in ("fixed_expert", "fixed_uniform", "recency", "autoregressive", "recurrent"):
            scores = [r["metrics"] for r in result["runs"] if r["condition"] == condition and r["family"] == family]
            summary[condition][family] = {k:float(np.mean([s[k] for s in scores if s[k] is not None]))
                                         if any(s[k] is not None for s in scores) else None for k in scores[0]}
    std = summary["standard"]
    ar = std["autoregressive"]
    gates = {"learning": ar["accuracy"] >= .70 and ar["accuracy"] >= std["fixed_uniform"]["accuracy"]+.10,
             "negative_control": .45 <= summary["random_labels"]["autoregressive"]["accuracy"] <= .55,
             "memory_dependence": ar["accuracy"] >= summary["no_feedback"]["autoregressive"]["accuracy"]+.10,
             "comparative": ar["accuracy"] >= std["recurrent"]["accuracy"]+.02 and ar["log_loss"] < std["recurrent"]["log_loss"]}
    paired = []
    for seed in TEST_SEEDS:
        get = lambda fam: np.mean([r["metrics"]["accuracy"] for r in result["runs"] if r["condition"]=="standard" and r["seed"]==seed and r["family"]==fam])
        paired.append(float(get("autoregressive")-get("recurrent")))
    result.update(summary=summary, gates=gates,
                  paired_ar_minus_recurrent={"per_episode_seed":paired,"mean":float(np.mean(paired)),
                                            "sample_sd":float(np.std(paired,ddof=1))})
    (output/"results.json").write_text(json.dumps(result, indent=2))
    lines = ["# Routing benchmark results", "", "Frozen weights; feedback updates state only. CPU NumPy; synthetic task.", "",
             "| Condition | Controller | Accuracy | Brier | Log loss | ECE | Confidently wrong | Sampled accuracy |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for condition, fams in summary.items():
        for family, m in fams.items():
            lines.append(f"| {condition} | {family} | {m['accuracy']:.3%} | {m['brier']:.4f} | {m['log_loss']:.4f} | {m['ece']:.4f} | {m['confidently_wrong_rate']:.3%} | {m['sampled_accuracy']:.3%} |")
    lines += ["", "## Preregistered scientific gates", ""]+[f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in gates.items()]
    lines += ["", "AR minus recurrent accuracy by independent test episode seed: "+str(paired),
              "", "Three parameter seeds share five test streams; 15 runs are not 15 independent environments.",
              "No hardware calibration, quantum advantage, AGI, or superiority outside this task is established.",
              "The expert bank and feedback features are hand designed and shared by all adaptive baselines."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
