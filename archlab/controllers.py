"""Tiny NumPy neural routers with analytic gradients and exact route likelihoods."""
import json
from pathlib import Path
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def responsibilities(q, outputs, label):
    likelihood = 0.05 + 0.9 * (outputs == label)
    probability = float(q @ likelihood)
    return q * likelihood / probability, -np.log(probability)


class Adam:
    def __init__(self, params, rate=0.003):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.rate = rate

    def step(self, params, grads):
        norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        scale = min(1.0, 5.0 / max(float(norm), 1e-12))
        self.t += 1
        for key, p in params.items():
            g = grads[key] * scale
            self.m[key] = .9*self.m[key] + .1*g
            self.v[key] = .999*self.v[key] + .001*g*g
            p -= self.rate * (self.m[key]/(1-.9**self.t)) / (np.sqrt(self.v[key]/(1-.999**self.t))+1e-8)


class NeuralBase:
    def reset(self):
        pass

    @property
    def parameter_count(self):
        return sum(v.size for v in self.params.values())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {**self.params, "state": getattr(self, "h", np.zeros(0))}
        np.savez(path, **state)
        path.with_suffix(".json").write_text(json.dumps({"kind": self.kind, "parameters": self.parameter_count,
                                                        "format": 1, "purpose": "inference checkpoint"}, indent=2))


class AutoregressiveRouter(NeuralBase):
    kind = "autoregressive"
    def __init__(self, seed=11, dim=24, hidden=24):
        r = np.random.default_rng(seed)
        self.params = {"W": r.normal(0, 1/np.sqrt(dim), (dim, hidden)), "b": np.zeros(hidden),
                       "A": r.normal(0, .1, (hidden, 2)), "a": np.zeros(2),
                       "B": r.normal(0, .1, (hidden, 4)), "b2": np.zeros(4)}

    def forward(self, x, temperature=1.0):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("positive finite temperature required")
        p = self.params
        h = np.tanh(x @ p["W"] + p["b"])
        pa = softmax((h @ p["A"] + p["a"]) / temperature)
        pb = softmax(((h @ p["B"] + p["b2"]) / temperature).reshape(2, 2))
        return (pa[:, None] * pb).ravel(), (x, h, pa, pb)

    def next_token(self, x, prefix=(), temperature=1.0):
        _, (_, _, pa, pb) = self.forward(x, temperature)
        if not prefix:
            return pa.copy()
        if len(prefix) == 1 and prefix[0] in (0, 1):
            return pb[prefix[0]].copy()
        raise ValueError("prefix must be empty or one binary token")

    def sample(self, x, rng, temperature=1.0, prefix=()):
        if prefix and (len(prefix) != 1 or prefix[0] not in (0, 1)):
            raise ValueError("invalid prefix")
        a = int(prefix[0]) if prefix else int(rng.choice(2, p=self.next_token(x, (), temperature)))
        b = int(rng.choice(2, p=self.next_token(x, (a,), temperature)))
        q, _ = self.forward(x, temperature)
        # Always return the ORIGINAL joint log likelihood, including forced prefix.
        return (a, b), float(np.log(q[2*a+b]))

    def gradients(self, caches, outputs, labels):
        g = {k: np.zeros_like(v) for k, v in self.params.items()}
        total = 0.0
        for cache, values, label in zip(caches, outputs, labels):
            x, h, pa, pb = cache
            q = (pa[:, None]*pb).ravel()
            resp, loss = responsibilities(q, values, label)
            resp = resp.reshape(2, 2)
            da = pa - resp.sum(axis=1)
            db = (resp.sum(axis=1)[:, None]*pb-resp).ravel()
            g["A"] += np.outer(h, da)
            g["a"] += da
            g["B"] += np.outer(h, db)
            g["b2"] += db
            dz = (da @ self.params["A"].T + db @ self.params["B"].T) * (1-h*h)
            g["W"] += np.outer(x, dz)
            g["b"] += dz
            total += loss
        n = len(caches)
        return {k:v/n for k,v in g.items()}, total/n


class RecurrentRouter(NeuralBase):
    kind = "recurrent"
    def __init__(self, seed=11, dim=24, hidden=24):
        r = np.random.default_rng(seed)
        self.params = {"W": r.normal(0, 1/np.sqrt(dim), (dim, hidden)),
                       "R": r.normal(0, .2/np.sqrt(hidden), (hidden, hidden)),
                       "b": np.zeros(hidden), "O": r.normal(0, .1, (hidden, 4)), "o": np.zeros(4)}
        self.reset()

    def reset(self):
        self.h = np.zeros(self.params["b"].size)

    def forward(self, x):
        p = self.params
        prev = self.h.copy()
        self.h = np.tanh(x @ p["W"] + prev @ p["R"] + p["b"])
        q = softmax(self.h @ p["O"] + p["o"])
        return q, (x, prev, self.h.copy(), q)

    def gradients(self, caches, outputs, labels):
        g = {k: np.zeros_like(v) for k,v in self.params.items()}
        carry = np.zeros_like(self.h)
        total = 0.0
        for (x, prev, h, q), values, label in reversed(list(zip(caches, outputs, labels))):
            resp, loss = responsibilities(q, values, label)
            d = q-resp
            g["O"] += np.outer(h, d)
            g["o"] += d
            dz = (d @ self.params["O"].T + carry) * (1-h*h)
            g["W"] += np.outer(x, dz)
            g["R"] += np.outer(prev, dz)
            g["b"] += dz
            carry = dz @ self.params["R"].T
            total += loss
        n = len(caches)
        return {k:v/n for k,v in g.items()}, total/n


class JointRouter(NeuralBase):
    """Conventional feedforward router; four routes in one categorical head."""
    kind = "joint"

    def __init__(self, seed=11, dim=24, hidden=26):
        r = np.random.default_rng(seed)
        self.params = {"W": r.normal(0, 1/np.sqrt(dim), (dim, hidden)),
                       "b": np.zeros(hidden),
                       "O": r.normal(0, .1, (hidden, 4)), "o": np.zeros(4)}

    def forward(self, x):
        p = self.params
        h = np.tanh(x @ p["W"] + p["b"])
        q = softmax(h @ p["O"] + p["o"])
        return q, (x, h, q)

    def gradients(self, caches, outputs, labels):
        g = {k:np.zeros_like(v) for k,v in self.params.items()}
        total = 0.0
        for (x, h, q), values, label in zip(caches, outputs, labels):
            resp, loss = responsibilities(q, values, label)
            d = q-resp
            g["O"] += np.outer(h, d)
            g["o"] += d
            dz = (d @ self.params["O"].T) * (1-h*h)
            g["W"] += np.outer(x, dz)
            g["b"] += dz
            total += loss
        n = len(caches)
        return {k:v/n for k,v in g.items()}, total/n


def load_model(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    classes = {"autoregressive": AutoregressiveRouter, "recurrent": RecurrentRouter,
               "joint": JointRouter}
    if meta["format"] != 1 or meta["kind"] not in classes:
        raise ValueError("unsupported checkpoint")
    with np.load(path, allow_pickle=False) as data:
        dim, hidden = data["W"].shape
        model = classes[meta["kind"]](dim=dim, hidden=hidden)
        for k, v in model.params.items():
            if data[k].shape != v.shape or not np.isfinite(data[k]).all():
                raise ValueError("invalid checkpoint parameter")
            model.params[k] = data[k].copy()
        if model.kind == "recurrent":
            if data["state"].shape != model.h.shape or not np.isfinite(data["state"]).all():
                raise ValueError("invalid recurrent state")
            model.h = data["state"].copy()
    return model
