"""Public inference and post-feedback mixture updates; no evaluator imports."""
import math
from confidence_lab.policy import ConfidenceState


class OnlineMixture:
    def __init__(self, prior, share, learning_rate, log_odds_clip=20.):
        if not (0<prior<1 and 0<=share<=1 and math.isfinite(learning_rate) and
                learning_rate>0 and math.isfinite(log_odds_clip) and log_odds_clip>0):
            raise ValueError("valid prior, sharing and learning rate required")
        self.prior, self.share = float(prior), float(share)
        self.learning_rate, self.clip = float(learning_rate),float(log_odds_clip)
        self.alpha = self.prior

    def observe(self, base, label):
        if not (.0<base<1. and label in (0,1)):
            raise ValueError("probability and released binary label required")
        likelihood = base if label else 1-base
        z = math.log(self.alpha)-math.log1p(-self.alpha)+self.learning_rate*math.log(likelihood/.5)
        posterior = 1/(1+math.exp(-max(-self.clip,min(self.clip,z))))
        self.alpha = (1-self.share)*posterior+self.share*self.prior


class Policy:
    def __init__(self, family, gate, selection, history_rates=(.04,.15)):
        if family not in ("uniform","recency","fixed_shrinkage","frozen_gate","online_mixture"):
            raise ValueError(family)
        self.family, self.gate, self.selection = family,gate,selection
        self.history = ConfidenceState(history_rates)
        s = selection["online"]
        self.online = OnlineMixture(s["prior"],s["share"],s["learning_rate"],s["log_odds_clip"])
        self.pending = None

    def predict(self, observation):
        base = self.history.base(observation)
        features = self.history.features(base)
        if self.family=="frozen_gate":
            alpha = self.gate.alpha(features)
        elif self.family=="online_mixture":
            alpha = self.online.alpha
        elif self.family=="fixed_shrinkage":
            alpha = self.selection["fixed_alpha"]
        else:
            alpha = 1. if self.family=="recency" else 0.
        self.pending = (observation,base)
        return {"base_probability":base,"alpha":alpha,
                "probability":float(.5+alpha*(base-.5))},features

    def observe(self, observation, label):
        if self.pending is None or self.pending[0]!=observation:
            raise RuntimeError("predict this observation before releasing feedback")
        self.online.observe(self.pending[1],label)
        self.history.observe(observation,label)
        self.pending = None
