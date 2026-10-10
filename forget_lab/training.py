"""Conventional recurrent baseline trained on stable AND changing streams."""
import hashlib
import numpy as np
from archlab.controllers import Adam,RecurrentRouter
from sensor_lab.public import Observation as SensorObservation
from recall_recovery.public import History
from recall_recovery.training import gradients,clone_router
from .world import episode


def train_router(experts,seed,cfg):
    router = RecurrentRouter(seed=seed,dim=14,hidden=cfg['recurrent_hidden'])
    optimizer = Adam(router.params,rate=cfg['recurrent_learning_rate'])
    rng = np.random.default_rng(cfg['training_shuffle_seed'])
    assert len(cfg['training_cases'])==len(cfg['training_seeds'])
    worlds = {s:episode(s,cfg['training_steps'],c,cfg) for s,c in zip(cfg['training_seeds'],cfg['training_cases'])}
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
        'parameters':router.parameter_count,'input_sha256':digest.hexdigest(),'training_cases':cfg['training_cases']}
