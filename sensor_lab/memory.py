"""Strict continuous-observation feedback history; no private metadata columns."""
import json
from pathlib import Path
import sqlite3
from .public import Observation


class Memory:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS feedback(namespace TEXT NOT NULL, step INTEGER NOT NULL, readings TEXT NOT NULL, label INTEGER NOT NULL, PRIMARY KEY(namespace,step))')
        self.db.commit()

    def append(self,namespace,step,observation,label):
        if not namespace or step<0 or label not in (0,1) or not isinstance(observation,Observation):
            raise ValueError('Public released feedback required')
        self.db.execute('INSERT INTO feedback VALUES (?,?,?,?)',
                        (namespace,step,json.dumps(observation.readings),int(label)))

    def history(self,namespace,*,before_step):
        if not namespace or before_step<0:
            raise ValueError('Explicit namespace and nonnegative causal cutoff required')
        return [(s,Observation(tuple(json.loads(x))),y) for s,x,y in self.db.execute(
            'SELECT step,readings,label FROM feedback WHERE namespace=? AND step<? ORDER BY step',
            (namespace,before_step))]

    def close(self):
        self.db.commit()
        self.db.close()
