"""Conventional neural recurrent router, with continuous-expert BPTT likelihood."""
import hashlib
import numpy as np
from archlab.controllers import Adam,RecurrentRouter,load_model
from sensor_lab.public import Observation as SensorObservation
from .public import History
from .world import episode


def gradients(router,caches,outputs,labels):
    g = {k:np.zeros_like(v) for k,v in router.params.items()}
    carry,total = np.zeros_like(router.h),0.
    for (x,previous,h,q),values,label in reversed(list(zip(caches,outputs,labels))):
        likelihood = .05+.9*(values if label else 1-values)
        probability = float(q@likelihood)
        d = q-q*likelihood/probability
        g['O'] += np.outer(h,d)
        g['o'] += d
        dz = (d@router.params['O'].T+carry)*(1-h*h)
        g['W'] += np.outer(x,dz)
        g['R'] += np.outer(previous,dz)
        g['b'] += dz
        carry = dz@router.params['R'].T
        total -= np.log(probability)
    n = len(caches)
    return {k:v/n for k,v in g.items()},total/n


def train_router(experts,seed,cfg):
    router = RecurrentRouter(seed=seed,dim=14,hidden=cfg['recurrent_hidden'])
    optimizer = Adam(router.params,rate=cfg['recurrent_learning_rate'])
    rng = np.random.default_rng(cfg['training_shuffle_seed'])
    worlds = {s:episode(s,cfg['training_steps'],'recurring',cfg) for s in cfg['training_seeds']}
    seen,updates,history = 0,0,[]
    digest = hashlib.sha256()
    for _ in range(cfg['recurrent_epochs']):
        epoch_losses = []
        for s in rng.permutation(cfg['training_seeds']):
            router.reset()
            state = History()
            caches,values,labels = [],[],[]
            for index,row in enumerate(worlds[int(s)]):
                x = state.router_input(row.observation)
                outputs = experts.predict(SensorObservation(row.observation.readings))
                q,cache = router.forward(x)
                base = float(.05+.9*(q@outputs))
                digest.update(np.asarray(x,dtype='<f8').tobytes())
                caches.append(cache)
                values.append(outputs)
                labels.append(row.label)
                state.observe(outputs,base,row.label)
                seen += 1
                if len(caches)==cfg['bptt_steps'] or index==len(worlds[int(s)])-1:
                    g,loss = gradients(router,caches,values,labels)
                    optimizer.step(router.params,g)
                    updates += 1
                    epoch_losses.append(loss)
                    caches,values,labels = [],[],[]
        history.append(float(np.mean(epoch_losses)))
    router.reset()
    return router,{'seed':seed,'loss_by_epoch':history,'examples_seen':seen,'updates':updates,
                   'parameters':router.parameter_count,'input_sha256':digest.hexdigest()}


def clone_router(router):
    copy = RecurrentRouter(dim=router.params['W'].shape[0],hidden=router.params['W'].shape[1])
    copy.params = {k:v.copy() for k,v in router.params.items()}
    for array in copy.params.values():
        array.setflags(write=False)
    copy.reset()
    return copy
