"""Private generator: cue identities and task mappings are never policy inputs."""
from dataclasses import dataclass
import numpy as np
from confidence_lab.policy import sigmoid
from .public import Observation


@dataclass(frozen=True,slots=True)
class PrivateRow:
    observation: Observation
    label: int
    context_id: int
    age: int
    switched: bool
    returned: bool
    mapping_changed: bool


def coefficients(cfg):
    rng = np.random.default_rng(cfg['coefficient_seed'])
    q,_ = np.linalg.qr(rng.normal(size=(8,4)))
    return q.T*cfg['coefficient_norm']


def calibration_data(cfg):
    xs,ys = [],[]
    coef = coefficients(cfg)
    for seed in cfg['calibration_seeds']:
        xr,yr = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        x = xr.normal(size=(cfg['calibration_steps'],8))
        labels = (yr.random((len(x),4))<sigmoid(x@coef.T)).astype(float)
        xs.append(x)
        ys.append(labels)
    return np.concatenate(xs),np.concatenate(ys)


def episode(seed,steps,case,cfg):
    if case not in cfg['cases']:
        raise ValueError(case)
    kr,xr,yr,sr,mr,nr = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    # New cue templates and mappings in every stream, independent of labels/readings.
    count = steps//cfg['block_interval'][0]+2 if case=='novel_contexts' else 4
    keys = []
    for _ in range(count):
        for attempt in range(10000):
            key = kr.normal(size=6)
            key *= cfg['cue_norm']/np.linalg.norm(key)
            if not keys or min(np.linalg.norm(key-k) for k in keys)>=cfg['minimum_cue_separation']:
                keys.append(key)
                break
        else:
            raise RuntimeError('Could not generate separated public cue templates')
    mapping = mr.integers(4,size=count) if case=='novel_contexts' else mr.permutation(4)
    readings,uniforms = xr.normal(size=(steps,8)),yr.random(steps)
    jitter = nr.normal(size=(steps,6))
    noise = cfg['noisy_cue_sigma'] if case=='noisy_cues' else cfg['cue_sigma']
    coef = coefficients(cfg)
    order = list(sr.permutation(4))
    block,index,age,remaining = 0,0,0,0
    seen = set()
    changed = False
    rows = []
    for step,x in enumerate(readings):
        switched = False
        if remaining==0:
            if block>0:
                switched = True
                block += 1
                index += 1
            else:
                block = 1
            if case=='novel_contexts':
                context = block-1
            else:
                if index==len(order):
                    previous = order[-1]
                    order = list(sr.permutation(4))
                    if order[0]==previous:
                        order[0],order[1] = order[1],order[0]
                    index = 0
                context = int(order[index])
            returned = context in seen
            seen.add(context)
            if case=='mapping_change' and step>=steps//2 and not changed:
                mapping = (mapping+1)%4
                changed = True
            remaining = int(sr.integers(cfg['block_interval'][0],cfg['block_interval'][1]+1))
            age = 0
        cue = keys[context]+noise*jitter[step]
        probability = .5 if case=='random_labels' else float(sigmoid(x@coef[mapping[context]]))
        rows.append(PrivateRow(Observation(tuple(float(v) for v in cue),tuple(float(v) for v in x)),
                               int(uniforms[step]<probability),context,age,switched,returned,changed))
        age += 1
        remaining -= 1
    return rows
