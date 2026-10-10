"""Scoped SQLite feedback memory with a mandatory causal retrieval cutoff."""
import json
from pathlib import Path
import sqlite3
import numpy as np
from .embedding import HashEmbedder


class Memory:
    def __init__(self, path, embedder=None):
        self.embedder = embedder or HashEmbedder()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS feedback (
            namespace TEXT NOT NULL, step INTEGER NOT NULL, text TEXT NOT NULL,
            observation TEXT NOT NULL, label INTEGER NOT NULL, embedding TEXT NOT NULL,
            model_id TEXT NOT NULL, PRIMARY KEY(namespace,step))""")
        self.db.commit()

    def append(self, namespace, step, observation, label):
        if not namespace or step < 0 or label not in (0, 1):
            raise ValueError("invalid feedback")
        if len(observation) != 2 or any(x not in (0, 1) for x in observation):
            raise ValueError("public observation must contain two bits")
        # The schema cannot accept hidden rule IDs, private state or arbitrary metadata.
        text = f"bit0_{observation[0]} bit1_{observation[1]}"
        vector = self.embedder.encode([text])[0].tolist()
        self.db.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?)",
                        (namespace, step, text, json.dumps(list(observation)), int(label),
                         json.dumps(vector), self.embedder.model_id))
        self.db.commit()

    def history(self, namespace, *, before_step):
        rows = self.db.execute("SELECT step,observation,label FROM feedback WHERE namespace=? "
                               "AND step<? ORDER BY step", (namespace, before_step)).fetchall()
        return [(t, tuple(json.loads(x)), y) for t, x, y in rows]

    def search(self, namespace, text, *, before_step, limit=5):
        if limit < 0:
            raise ValueError("negative limit")
        q = self.embedder.encode([text])[0]
        rows = self.db.execute("SELECT step,text,embedding,model_id FROM feedback "
                               "WHERE namespace=? AND step<?", (namespace, before_step)).fetchall()
        scored = []
        for t, txt, emb, model in rows:
            if model != self.embedder.model_id:
                raise ValueError("embedding version mismatch; explicitly rebuild memory")
            scored.append({"step": t, "text": txt, "score": float(q @ np.array(json.loads(emb)))})
        return sorted(scored, key=lambda x: (-x["score"], -x["step"]))[:limit]

    def close(self):
        self.db.close()
