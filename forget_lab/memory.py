"""Causal fading changes retrieval weights; the complete audit history remains."""
import numpy as np
from recall_recovery.memory import Memory as OriginalMemory


class Memory(OriginalMemory):
    def recent(self,namespace,observation,*,before_step,capacity,limit,radius,half_life,permutation=None):
        if not namespace or before_step<0 or capacity<1 or limit<1 or radius<=0 or (half_life is not None and half_life<=0):
            raise ValueError('Positive retrieval budgets and causal cutoff required')
        self._load(namespace)
        if namespace not in self.arrays:
            return None,0,[],0.
        a = self.arrays[namespace]
        end = int(np.searchsorted(a['steps'][:a['count']],before_step,side='left'))
        start = max(0,end-capacity)
        keys = a['keys'][start:end]
        if permutation is not None:
            keys = keys[permutation.permutation(len(keys))]
        query = self.embedder.encode([observation])[0]
        distances = np.sum((keys-query)**2,axis=1)
        chosen = np.flatnonzero(distances<=radius*radius)[-limit:]
        if not len(chosen):
            return None,0,[],0.
        # Rank is among eligible matching feedback, never a hidden context age.
        ranks = np.arange(len(chosen)-1,-1,-1,dtype=float)
        weights = np.ones(len(chosen)) if half_life is None else np.exp2(-ranks/half_life)
        weights /= weights.sum()
        errors = a['errors'][start:end][chosen]
        return weights@errors,len(chosen),a['steps'][start:end][chosen].tolist(),float(1/(weights@weights))
