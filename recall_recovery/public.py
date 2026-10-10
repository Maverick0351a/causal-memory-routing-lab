"""Observable cues, learned experts and feedback summaries; no world imports."""
from dataclasses import dataclass
import numpy as np
from archlab.controllers import Adam
from confidence_lab.policy import sigmoid
from sensor_lab.public import Ensemble,logistic_objective


@dataclass(frozen=True,slots=True)
class Observation:
    cue: tuple[float,...]
    readings: tuple[float,...]

    def __post_init__(self):
        if len(self.cue)!=6 or len(self.readings)!=8 or not np.isfinite((*self.cue,*self.readings)).all():
            raise ValueError('Six finite public cue values and eight finite readings required')


class CueEmbedder:
    """Identity encoding of measured continuous cues, not semantic embeddings."""
    dimension = 6
    model_id = 'continuous-public-cue-v1'

    def encode(self,observations):
        return np.array([o.cue for o in observations],dtype=float).reshape(-1,6)


def fit_experts(x,labels,cfg):
    design = np.column_stack((np.ones(len(x)),x))
    weights,histories = [],[]
    for index,seed in enumerate(cfg['expert_seeds']):
        take = np.random.default_rng(seed).integers(len(x),size=len(x))
        a,b = design[take],labels[take,index]
        params = {'weights':np.zeros(9)}
        optimizer = Adam(params,rate=cfg['expert_learning_rate'])
        history = []
        for _ in range(cfg['expert_epochs']):
            loss,gradient = logistic_objective(params['weights'],a,b,cfg['expert_l2'])
            optimizer.step(params,{'weights':gradient})
            history.append(loss)
        weights.append(params['weights'].copy())
        histories.append(history)
    return Ensemble(weights),histories


class History:
    def __init__(self):
        self.fast = np.full(4,.5)
        self.slow = np.full(4,.5)
        self.correct = np.full(2,.5)
        self.gain = np.zeros(2)
        self.rates = np.array([.04,.15])

    def router_input(self,observation):
        return np.array([*(np.array(observation.cue)/3.),*self.fast-.5,*self.slow-.5])

    def gate_input(self,base):
        return np.array([1.,abs(2*base-1),*self.correct-.5,*self.gain,
                         .5-self.fast.min(),.5-self.slow.min()])

    def observe(self,outputs,base,label):
        error = ((outputs>=.5)!=label).astype(float)
        self.fast += .35*(error-self.fast)
        self.slow += .04*(error-self.slow)
        self.correct += self.rates*(float((base>=.5)==label)-self.correct)
        self.gain += self.rates*(np.log((base if label else 1-base)/.5)-self.gain)
        return error
