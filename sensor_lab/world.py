"""Evaluator-private synthetic binary fault propensity, not physical sensor data."""
from dataclasses import dataclass
import numpy as np
from confidence_lab.policy import sigmoid
from .public import Observation


@dataclass(frozen=True, slots=True)
class PrivateRow:
    observation: Observation
    label: int
    mode: str
    age: int
    switched: bool


def episode(seed, steps, case, cfg):
    if case not in cfg['cases']:
        raise ValueError(case)
    xr,yr,sr,wr = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    readings,uniforms = xr.normal(size=(steps,8)),yr.random(steps)
    original = np.array(cfg['sensor_coefficients'],dtype=float)
    weights = original.copy()
    mixed = case in ('mixed','short_mixed','no_feedback')
    mode = ('signal' if sr.integers(2) else 'noise') if mixed else 'noise' if case=='random_labels' else 'signal'
    interval = cfg['short_interval'] if case=='short_mixed' else cfg['mode_interval']
    remaining = int(sr.integers(interval[0],interval[1]+1))
    age,rows = 0,[]
    for step,x in enumerate(readings):
        switched = False
        if remaining==0:
            if mixed:
                mode = 'noise' if mode=='signal' else 'signal'
            elif case=='rotating':
                weights = wr.normal(size=8)
                weights *= np.linalg.norm(original)/np.linalg.norm(weights)
            switched = mixed or case=='rotating'
            remaining = int(sr.integers(interval[0],interval[1]+1))
            if switched:
                age = 0
        gain = cfg['weak_signal_gain'] if case=='weak_signal' else 1.
        probability = .5 if mode=='noise' else float(sigmoid(gain*(x@weights)))
        label = int(uniforms[step]<probability)
        rows.append(PrivateRow(Observation(tuple(float(v) for v in x)),label,mode,age,switched))
        remaining -= 1
        age += 1
    return rows


def training_data(cfg):
    rows = [r for seed in cfg['training_seeds'] for r in episode(seed,cfg['train_steps'],'stable',cfg)]
    x = np.array([r.observation.readings for r in rows])
    y = np.array([r.label for r in rows],dtype=float)
    return x,y
