"""Public-only inference. No benchmark/evaluator imports or hidden-state inputs."""
from pathlib import Path
import numpy as np
from archlab.controllers import Adam, softmax
from archlab.public import FeedbackState, experts


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -20, 20)))


class ConfidenceState:
    def __init__(self, rates=(.04, .15)):
        self.feedback = FeedbackState()
        self.rates = np.asarray(rates, dtype=float)
        self.correct = np.full(2, .5)
        self.log_gain = np.zeros(2)

    def base(self, observation):
        q = softmax(-8*self.feedback.fast)
        return float(.05 + .9*(q @ experts(observation)))

    def features(self, base):
        return np.array([1., abs(2*base-1), *self.correct-.5, *self.log_gain,
                         .5-self.feedback.fast.min(), .5-self.feedback.slow.min()])

    def observe(self, observation, label):
        # Reconstruct the prior prediction before updating any feedback statistic.
        p = self.base(observation)
        correct = float((p >= .5) == label)
        gain = np.log(p if label else 1-p) - np.log(.5)
        self.correct += self.rates*(correct-self.correct)
        self.log_gain += self.rates*(gain-self.log_gain)
        self.feedback.observe(observation, label)


class Gate:
    def __init__(self, weights, mean, scale):
        self.weights, self.mean, self.scale = [np.array(a, dtype=float, copy=True)
                                              for a in (weights, mean, scale)]
        if any(a.shape != (8,) or not np.isfinite(a).all()
               for a in (self.weights, self.mean, self.scale)) or (self.scale <= 0).any():
            raise ValueError("eight finite coefficients with positive scales required")
        for array in (self.weights, self.mean, self.scale):
            array.setflags(write=False)

    def alpha(self, features):
        return float(sigmoid(((features-self.mean)/self.scale) @ self.weights))

    def save(self, path):
        np.savez(path, weights=self.weights, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as saved:
            return cls(saved["weights"], saved["mean"], saved["scale"])


def objective(weights, x, base, y, penalty):
    raw = x @ weights
    alpha = sigmoid(raw)
    p = .5+alpha*(base-.5)
    loss = np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))
    derivative = (p-y)/(p*(1-p))*(base-.5)*alpha*(1-alpha)*(np.abs(raw)<20)
    regularized = weights.copy()
    regularized[0] = 0
    return (float(loss+.5*penalty*(regularized@regularized)),
            x.T@derivative/len(y)+penalty*regularized)


def fit(x, base, y, penalty, cfg):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), cfg["feature_scale_floor"])
    mean[0], scale[0] = 0, 1
    standardized = (x-mean)/scale
    parameters = {"weights":np.zeros(x.shape[1])}
    optimizer = Adam(parameters, rate=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["shuffle_seed"])
    losses = []
    for _ in range(cfg["epochs"]):
        order = rng.permutation(len(y))
        for start in range(0, len(y), cfg["batch_size"]):
            idx = order[start:start+cfg["batch_size"]]
            _, gradient = objective(parameters["weights"],standardized[idx],base[idx],y[idx],penalty)
            optimizer.step(parameters,{"weights":gradient})
        losses.append(objective(parameters["weights"],standardized,base,y,penalty)[0])
    return Gate(parameters["weights"],mean,scale), {
        "penalty":penalty,"examples":len(y),"epochs":cfg["epochs"],
        "optimizer_updates":optimizer.t,"regularized_loss_by_epoch":losses}
