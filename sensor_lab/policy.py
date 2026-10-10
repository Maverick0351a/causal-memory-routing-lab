"""Calibration receives only current public predictions and past labels."""
import math
import numpy as np
from confidence_lab.policy import sigmoid

FAMILIES = ('uniform','base','fixed_calibration','frozen_gate','online_calibration')


def logit(p):
    if not 0<p<1:
        raise ValueError('Interior probability required')
    return math.log(p)-math.log1p(-p)


class OnlineCalibration:
    """Two-parameter Platt calibration, direct causal log-loss SGD."""
    def __init__(self, initial, rate, decay):
        self.initial = np.array(initial,dtype=float,copy=True)
        if self.initial.shape!=(2,) or not np.isfinite(self.initial).all() or rate<=0 or decay<0:
            raise ValueError('Finite slope/intercept and valid rate/decay required')
        self.weights = self.initial.copy()
        self.rate,self.decay = rate,decay

    def predict(self, base):
        return float(sigmoid(self.weights @ np.array([logit(base),1.])))

    def observe(self, base, label):
        if label not in (0,1):
            raise ValueError('Released binary label required')
        x = np.array([logit(base),1.])
        gradient = (self.predict(base)-label)*x+self.decay*(self.weights-self.initial)
        self.weights = np.clip(self.weights-self.rate*gradient,-4.,4.)


class Policy:
    def __init__(self, family, gate, selection):
        if family not in FAMILIES:
            raise ValueError(family)
        self.family,self.gate = family,gate
        self.fixed = np.array(selection['fixed_weights'],dtype=float)
        self.online = OnlineCalibration(self.fixed,selection['online']['rate'],selection['online']['decay'])
        self.pending = None

    def predict(self, base, features):
        if self.pending is not None:
            raise RuntimeError('Feedback or discard required before next prediction')
        alpha,slope,intercept = None,None,None
        if self.family=='frozen_gate':
            alpha = self.gate.alpha(features)
            p = .5+alpha*(base-.5)
        elif self.family in ('fixed_calibration','online_calibration'):
            weights = self.fixed if self.family=='fixed_calibration' else self.online.weights
            slope,intercept = weights.tolist()
            p = float(sigmoid(slope*logit(base)+intercept))
        else:
            p = .5 if self.family=='uniform' else base
        self.pending = base
        return {'probability':p,'base_probability':base,'alpha':alpha,'slope':slope,'intercept':intercept}

    def observe(self, label):
        if self.pending is None:
            raise RuntimeError('Prediction required before feedback')
        if label not in (0,1):
            raise ValueError('Released binary label required')
        if self.family=='online_calibration':
            self.online.observe(self.pending,label)
        self.pending = None

    def discard(self):
        self.pending = None
