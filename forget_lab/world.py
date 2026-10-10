"""Private evaluation worlds: changes are never exposed to a controller."""
from dataclasses import dataclass
import numpy as np
from confidence_lab.policy import sigmoid
from recall_recovery.public import Observation
from recall_recovery.world import coefficients,calibration_data


@dataclass(frozen=True,slots=True)
class PrivateRow:
    observation: Observation
    label: int
    context_id: int
    age: int
    switched: bool
    returned: bool
    mapping_changed: bool
    revision: int
    changed_entry: bool


def episode(seed,steps,case,cfg):
    if case not in cfg['cases']:
        raise ValueError(case)
    kr,xr,yr,sr,mr,nr = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
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
            raise RuntimeError('Could not generate separated cues')
    mapping = mr.integers(4,size=count) if case=='novel_contexts' else mr.permutation(4)
    readings,uniforms = xr.normal(size=(steps,8)),yr.random(steps)
    jitter = nr.normal(size=(steps,6))
    noise = cfg['noisy_cue_sigma'] if case in ('noisy_cues','noisy_changes') else cfg['cue_sigma']
    coef = coefficients(cfg)
    order = list(sr.permutation(4))
    block,index,age,remaining,revision = 0,0,0,0,0
    seen,last_revision = set(),{}
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
            should_change = (case=='mapping_change' and step>=steps//2 and revision==0)
            should_change |= (case in ('repeated_changes','noisy_changes') and block>cfg['change_cycle_blocks']
                              and (block-1)%cfg['change_cycle_blocks']==0)
            if should_change:
                mapping = (mapping+1)%4
                revision += 1
            returned = context in seen
            changed_entry = revision>0 and last_revision.get(context,-1)<revision
            seen.add(context)
            last_revision[context] = revision
            remaining = int(sr.integers(cfg['block_interval'][0],cfg['block_interval'][1]+1))
            age = 0
        cue = keys[context]+noise*jitter[step]
        probability = .5 if case=='random_labels' else float(sigmoid(x@coef[mapping[context]]))
        rows.append(PrivateRow(Observation(tuple(float(v) for v in cue),tuple(float(v) for v in x)),
            int(uniforms[step]<probability),context,age,switched,returned,revision>0,revision,changed_entry))
        age += 1
        remaining -= 1
    return rows
