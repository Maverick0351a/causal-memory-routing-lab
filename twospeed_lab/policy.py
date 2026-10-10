"""Disjoint stable/fast views and contextual prequential selection.

Only Observation enters predict. Selector scores are updated with probabilities
committed before the current label. Private world metadata is never imported.
"""
import numpy as np
from archlab.controllers import softmax
from sensor_lab.public import Observation as SensorObservation
from recall_recovery.policy import Policy as BaselinePolicy

FAMILIES = ('two_speed', 'stable_only', 'fast_only', 'recent_uniform', 'fading',
            'adaptive_window', 'shuffled_two_speed', 'recency', 'context_cache',
            'recurrent', 'fixed_uniform')
NEW = FAMILIES[:7]
POLICY_KEYS = ('shuffled_memory_seed', 'routing_temperature', 'evidence_warmup',
               'retrieval_radius', 'selector_capacity', 'fast_prior_penalty',
               'memory_capacity', 'fast_capacity', 'neighbors', 'fast_neighbors',
               'fast_half_life', 'adaptive_windows')


def public_config(cfg):
    """Do not give controllers generator seeds, case labels or switch schedules."""
    return {key: cfg[key] for key in POLICY_KEYS}


class Selector:
    """Bounded cue prototypes with EMA losses for already committed forecasts."""
    def __init__(self, size, cfg, setting):
        self.size, self.cfg, self.setting = size, cfg, setting
        self.entries = []

    def find(self, cue):
        distances = [np.linalg.norm(np.array(cue)-r['key']) for r in self.entries]
        index = int(np.argmin(distances)) if distances else -1
        return index if index >= 0 and distances[index] <= self.cfg['retrieval_radius'] else None

    def weights(self, cue, available):
        index = self.find(cue)
        loss = self.entries[index]['loss'] if index is not None else np.full(self.size, np.log(2.))
        penalty = np.zeros(self.size)
        if self.size == 2:
            penalty[1] = self.cfg['fast_prior_penalty']
        logits = -self.setting['temperature']*(loss+penalty)
        logits = np.where(available, logits, -1e6)
        return softmax(logits) if any(available) else np.full(self.size, 1/self.size)

    def observe(self, cue, predictions, label):
        index = self.find(cue)
        if index is None:
            if len(self.entries) == self.cfg['selector_capacity']:
                self.entries.pop(0)
            self.entries.append({'key': np.array(cue, dtype=float), 'loss': np.full(self.size, np.log(2.))})
            index = len(self.entries)-1
        row = self.entries.pop(index)
        losses = -np.log(np.where(label, predictions, 1-predictions))
        row['loss'] += self.setting['rate']*(losses-row['loss'])
        row['key'] += .05*(np.array(cue)-row['key'])
        self.entries.append(row)


class Policy(BaselinePolicy):
    def __init__(self, family, experts, gate, selection, cfg, store, router=None, namespace='episode'):
        if family not in FAMILIES:
            raise ValueError(family)
        super().__init__('recency' if family in NEW else family, experts, gate, selection, cfg, store, router, namespace)
        self.family = family
        setting = selection['adaptive_window' if family == 'adaptive_window' else 'two_speed']
        self.selector = Selector(3 if family == 'adaptive_window' else 2, cfg, setting)
        self.branches = None

    def candidates(self, observation):
        self.store._load(self.namespace)
        if self.namespace not in self.store.arrays:
            return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0, dtype=bool), 0
        a = self.store.arrays[self.namespace]
        end = int(np.searchsorted(a['steps'][:a['count']], self.step, side='left'))
        start = max(0, end-self.cfg['memory_capacity'])
        keys = a['keys'][start:end]
        if self.family == 'shuffled_two_speed':
            keys = keys[self.shuffle.permutation(len(keys))]
        distances = np.sum((keys-np.array(observation.cue))**2, axis=1)
        take = np.flatnonzero(distances <= self.cfg['retrieval_radius']**2)
        # Stable = older 192; fast = latest 64 released records. No overlap.
        fast = take >= max(0, end-start-self.cfg['fast_capacity'])
        return a['errors'][start:end][take], a['steps'][start:end][take], fast, end-start

    def route(self, errors, half_life=None):
        count = len(errors)
        loss = self.history.fast.copy()
        if count:
            ranks = np.arange(count-1, -1, -1, dtype=float)
            w = np.ones(count) if half_life is None else np.exp2(-ranks/half_life)
            w /= w.sum()
            blend = min(1., count/self.cfg['evidence_warmup'])
            loss = (1-blend)*loss+blend*(w@errors)
        return softmax(-self.cfg['routing_temperature']*loss)

    def predict(self, observation):
        if self.family not in NEW:
            pred, features = super().predict(observation)
            pred.update(selector_weights=[], scanned_records=0, stable_count=0, fast_count=0)
            return pred, features
        if self.pending is not None:
            raise RuntimeError('Feedback or discard required before next prediction')
        outputs = self.experts.predict(SensorObservation(observation.readings))
        errors, steps, fast, scanned = self.candidates(observation)
        stable_errors = errors[~fast][-self.cfg['neighbors']:]
        fast_errors = errors[fast][-self.cfg['fast_neighbors']:]
        qs = [self.route(stable_errors), self.route(fast_errors, self.cfg['fast_half_life'])]
        available = [len(stable_errors)>0, len(fast_errors)>0]
        if self.family == 'adaptive_window':
            qs = [self.route(errors[-n:]) for n in self.cfg['adaptive_windows']]
            available = [len(errors)>0]*len(qs)
        probabilities = np.array([float(.05+.9*(q@outputs)) for q in qs])
        weights = self.selector.weights(observation.cue, available)
        if self.family == 'stable_only':
            weights = np.array([1., 0.])
        elif self.family == 'fast_only':
            weights = np.array([0., 1.])
        if self.family in ('recent_uniform', 'fading'):
            q = self.route(errors[-self.cfg['neighbors']:],
                           self.selection['fading']['half_life'] if self.family == 'fading' else None)
            weights = np.zeros(2)
        else:
            q = weights@np.asarray(qs)
        base = float(.05+.9*(q@outputs))
        features = self.history.gate_input(base)
        alpha = self.gate.alpha(features)
        self.pending = observation, outputs, base
        self.branches = probabilities.copy()
        return {'probability': base, 'gated_probability': float(.5+alpha*(base-.5)), 'alpha': alpha,
                'route': q.tolist(), 'retrieved_count': len(errors),
                'retrieved_max_step': int(max(steps)) if len(steps) else None,
                'selector_weights': weights.tolist(), 'scanned_records': scanned,
                'stable_count': len(stable_errors), 'fast_count': len(fast_errors)}, features

    def observe(self, observation, label):
        # Validate pending identity and label before mutating either state.
        if label not in (0, 1) or self.pending is None or self.pending[0] != observation:
            raise ValueError('Committed prediction and matching released feedback required')
        if self.family in NEW:
            self.selector.observe(observation.cue, self.branches, label)
        super().observe(observation, label)
        self.branches = None
