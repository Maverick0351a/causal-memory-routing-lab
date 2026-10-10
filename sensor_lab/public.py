"""Public sensor types and supervised predictor; no world/evaluator imports."""
from dataclasses import dataclass
import numpy as np
from archlab.controllers import Adam, softmax
from confidence_lab.policy import sigmoid


@dataclass(frozen=True, slots=True)
class Observation:
    readings: tuple[float, ...]

    def __post_init__(self):
        if len(self.readings) != 8 or not np.isfinite(self.readings).all():
            raise ValueError('Eight finite public readings required')


class Ensemble:
    def __init__(self, weights):
        self.weights = np.array(weights, dtype=float, copy=True)
        if self.weights.shape != (4,9) or not np.isfinite(self.weights).all():
            raise ValueError('Four nine-parameter sensor predictors required')
        self.weights.setflags(write=False)

    def predict(self, observation):
        return sigmoid(self.weights @ np.array((1., *observation.readings)))

    def save(self, path):
        np.savez(path, weights=self.weights)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['weights'])


def logistic_objective(weights, x, y, l2):
    z = x @ weights
    loss = np.mean(np.logaddexp(0,z)-y*z)
    regularized = weights.copy()
    regularized[0] = 0
    return float(loss+.5*l2*(regularized@regularized)), x.T@(sigmoid(z)-y)/len(y)+l2*regularized


def fit_ensemble(x, y, cfg):
    design = np.column_stack((np.ones(len(x)),x))
    weights, histories = [], []
    for seed in cfg['model_seeds']:
        indices = np.random.default_rng(seed).integers(len(y),size=len(y))
        a,b = design[indices], y[indices]
        params = {'weights':np.zeros(9)}
        adam = Adam(params,rate=cfg['model_learning_rate'])
        history = []
        for _ in range(cfg['model_epochs']):
            loss, gradient = logistic_objective(params['weights'],a,b,cfg['model_l2'])
            adam.step(params,{'weights':gradient})
            history.append(loss)
        weights.append(params['weights'].copy())
        histories.append(history)
    return Ensemble(weights), histories


class Adapter:
    """The original eight feature meanings, applied to learned sensor experts."""
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.fast = np.full(4,.5)
        self.slow = np.full(4,.5)
        self.correct = np.full(2,.5)
        self.gain = np.zeros(2)
        self.rates = np.array([.04,.15])
        self.pending = None

    def predict(self, observation):
        if self.pending is not None:
            raise RuntimeError('Release or explicitly discard pending feedback first')
        outputs = self.ensemble.predict(observation)
        base = float(.05+.9*(softmax(-8*self.fast)@outputs))
        features = np.array([1.,abs(2*base-1),*self.correct-.5,*self.gain,
                             .5-self.fast.min(),.5-self.slow.min()])
        self.pending = observation,base,outputs
        return base,features

    def observe(self, observation, label):
        if label not in (0,1):
            raise ValueError('Released binary label required')
        if self.pending is None or self.pending[0] != observation:
            raise RuntimeError('Predict this observation before feedback')
        _,base,outputs = self.pending
        correct = float((base>=.5)==label)
        gain = np.log(base if label else 1-base)-np.log(.5)
        self.correct += self.rates*(correct-self.correct)
        self.gain += self.rates*(gain-self.gain)
        mismatch = ((outputs>=.5)!=label).astype(float)
        self.fast += .35*(mismatch-self.fast)
        self.slow += .04*(mismatch-self.slow)
        self.pending = None

    def discard(self):
        self.pending = None
