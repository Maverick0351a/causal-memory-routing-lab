"""SQLite feedback plus causal RAM index; only public released feedback is stored."""
import json
from pathlib import Path
import sqlite3
import numpy as np
from .public import Observation,CueEmbedder


class Memory:
    def __init__(self,path=':memory:'):
        if str(path)!=':memory:':
            Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS feedback(namespace TEXT NOT NULL, step INTEGER NOT NULL, cue TEXT NOT NULL, readings TEXT NOT NULL, label INTEGER NOT NULL, errors TEXT NOT NULL, model_id TEXT NOT NULL, PRIMARY KEY(namespace,step))')
        self.db.commit()
        self.embedder = CueEmbedder()
        self.index = {}
        self.arrays = {}

    def _index_append(self,namespace,step,observation,errors):
        if namespace not in self.arrays:
            self.arrays[namespace] = {'steps':np.zeros(256,dtype=np.int64),
                'keys':np.zeros((256,6)),'errors':np.zeros((256,4)),'count':0}
        a = self.arrays[namespace]
        n = a['count']
        if n==len(a['steps']):
            for key in ('steps','keys','errors'):
                a[key] = np.concatenate((a[key],np.zeros_like(a[key])),axis=0)
        a['steps'][n],a['keys'][n],a['errors'][n] = step,observation.cue,errors
        a['count'] += 1

    def _load(self,namespace):
        if namespace not in self.index:
            rows = self.db.execute('SELECT step,cue,readings,label,errors,model_id FROM feedback WHERE namespace=? ORDER BY step',(namespace,)).fetchall()
            parsed = []
            for step,cue,readings,label,errors,model_id in rows:
                if model_id!=self.embedder.model_id:
                    raise ValueError('Memory embedding version changed')
                record = (step,Observation(tuple(json.loads(cue)),tuple(json.loads(readings))),label,np.array(json.loads(errors)))
                parsed.append(record)
                self._index_append(namespace,step,record[1],record[3])
            self.index[namespace] = parsed
        return self.index[namespace]

    def append(self,namespace,step,observation,label,errors):
        errors = np.asarray(errors,dtype=float)
        if not namespace or step<0 or label not in (0,1) or not isinstance(observation,Observation) or errors.shape!=(4,) or not np.isin(errors,[0.,1.]).all():
            raise ValueError('Public released feedback and four binary expert errors required')
        rows = self._load(namespace)
        if rows and step<=rows[-1][0]:
            raise ValueError('Append steps must increase within a namespace')
        self.db.execute('INSERT INTO feedback VALUES (?,?,?,?,?,?,?)',(namespace,step,
            json.dumps(observation.cue),json.dumps(observation.readings),int(label),json.dumps(errors.tolist()),self.embedder.model_id))
        rows.append((step,observation,int(label),errors.copy()))
        self._index_append(namespace,step,observation,errors)

    def history(self,namespace,*,before_step):
        if not namespace or before_step<0:
            raise ValueError('Explicit namespace and nonnegative causal cutoff required')
        return [(step,obs,label,errors.copy()) for step,obs,label,errors in self._load(namespace) if step<before_step]

    def search(self,namespace,observation,*,before_step,capacity,limit,radius,permutation=None):
        if not namespace or before_step<0 or capacity<1 or limit<1 or radius<=0:
            raise ValueError('Positive memory budget and radius required')
        self._load(namespace)
        if namespace not in self.arrays:
            return None,0,[]
        a = self.arrays[namespace]
        end = int(np.searchsorted(a['steps'][:a['count']],before_step,side='left'))
        start = max(0,end-capacity)
        if end==start:
            return None,0,[]
        keys = a['keys'][start:end]
        if permutation is not None:
            keys = keys[permutation.permutation(len(keys))]
        query = self.embedder.encode([observation])[0]
        distance = np.sum((keys-query)**2,axis=1)
        candidates = np.flatnonzero(distance<=radius*radius)
        if not len(candidates):
            return None,0,[]
        # Stable distance ties resolve by original order, using only past records.
        chosen = candidates[np.argsort(distance[candidates],kind='stable')[:limit]]
        errors = a['errors'][start:end][chosen]
        return errors.mean(axis=0),len(chosen),a['steps'][start:end][chosen].tolist()

    def close(self):
        self.db.commit()
        self.db.close()
