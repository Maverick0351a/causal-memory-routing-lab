"""Policy interfaces never receive private IDs, ages, changes or generator seeds."""
import numpy as np
from archlab.controllers import softmax
from sensor_lab.public import Observation as SensorObservation
from .public import History

FAMILIES = ('memory','memory_disabled','shuffled_memory','recency','recurrent','context_cache','fixed_uniform')


class ContextCache:
    def __init__(self,settings):
        self.settings = settings
        self.entries = []

    def query(self,observation):
        if not self.entries:
            return None,0
        distances = [np.linalg.norm(np.array(observation.cue)-e['key']) for e in self.entries]
        index = int(np.argmin(distances))
        entry = self.entries[index]
        return (entry['errors'].copy(),entry['count']) if distances[index]<=self.settings['radius'] else (None,0)

    def observe(self,observation,errors):
        key = np.array(observation.cue)
        distances = [np.linalg.norm(key-e['key']) for e in self.entries]
        index = int(np.argmin(distances)) if distances else -1
        if index<0 or distances[index]>self.settings['radius']:
            if len(self.entries)==self.settings['capacity']:
                self.entries.pop(0)
            self.entries.append({'key':key.copy(),'errors':np.full(4,.5),'count':0})
            index = len(self.entries)-1
        entry = self.entries.pop(index)
        entry['key'] += .05*(key-entry['key'])
        entry['errors'] += self.settings['rate']*(errors-entry['errors'])
        entry['count'] += 1
        self.entries.append(entry)


class Policy:
    def __init__(self,family,experts,gate,selection,cfg,store,router=None,namespace='episode'):
        if family not in FAMILIES:
            raise ValueError(family)
        self.family,self.experts,self.gate = family,experts,gate
        self.selection,self.cfg,self.store,self.namespace = selection,cfg,store,namespace
        self.history = History()
        self.router = router
        if self.router is not None:
            self.router.reset()
        self.cache = ContextCache(selection['cache'])
        self.shuffle = np.random.default_rng(cfg['shuffled_memory_seed'])
        self.step,self.pending = 0,None

    def predict(self,observation):
        if self.pending is not None:
            raise RuntimeError('Feedback or discard required before next prediction')
        outputs = self.experts.predict(SensorObservation(observation.readings))
        loss = self.history.fast.copy()
        count,steps = 0,[]
        if self.family in ('memory','shuffled_memory'):
            settings = self.selection['memory']
            retrieved,count,steps = self.store.search(self.namespace,observation,before_step=self.step,
                capacity=settings['capacity'],limit=settings['neighbors'],radius=settings['radius'],
                permutation=self.shuffle if self.family=='shuffled_memory' else None)
            if retrieved is not None:
                blend = settings['blend']*min(1.,count/self.cfg['evidence_warmup'])
                loss = (1-blend)*loss+blend*retrieved
        elif self.family=='context_cache':
            retrieved,count = self.cache.query(observation)
            if retrieved is not None:
                blend = self.selection['cache']['blend']*min(1.,count/self.cfg['evidence_warmup'])
                loss = (1-blend)*loss+blend*retrieved
        if self.family=='recurrent':
            q,_ = self.router.forward(self.history.router_input(observation))
            temperature = self.selection['recurrent']['temperature']
            q = softmax(np.log(q)/temperature)
        elif self.family=='fixed_uniform':
            q = np.full(4,.25)
        else:
            q = softmax(-self.cfg['routing_temperature']*loss)
        base = float(.05+.9*(q@outputs))
        features = self.history.gate_input(base)
        alpha = self.gate.alpha(features)
        self.pending = observation,outputs,base
        return {'probability':base,'gated_probability':float(.5+alpha*(base-.5)),
                'alpha':alpha,'route':q.tolist(),'retrieved_count':count,
                'retrieved_max_step':max(steps) if steps else None},features

    def observe(self,observation,label):
        if label not in (0,1):
            raise ValueError('Released binary label required')
        if self.pending is None or self.pending[0]!=observation:
            raise RuntimeError('Prediction for this observation required before feedback')
        _,outputs,base = self.pending
        errors = self.history.observe(outputs,base,label)
        if self.family=='context_cache':
            self.cache.observe(observation,errors)
        self.pending = None
        self.step += 1

    def discard(self):
        if self.pending is None:
            raise RuntimeError('Prediction required')
        self.pending = None
        self.step += 1
