"""Fading and matched controls; policy input remains a public Observation only."""
import numpy as np
from archlab.controllers import softmax
from sensor_lab.public import Observation as SensorObservation
from recall_recovery.policy import Policy as OriginalPolicy,FAMILIES as ORIGINAL_FAMILIES

NEW_FAMILIES = ('forgetting','recent_uniform','shuffled_forgetting')
FAMILIES = (*NEW_FAMILIES,*ORIGINAL_FAMILIES)


class Policy(OriginalPolicy):
    def __init__(self,family,experts,gate,selection,cfg,store,router=None,namespace='episode'):
        if family not in FAMILIES:
            raise ValueError(family)
        super().__init__('memory' if family in NEW_FAMILIES else family,experts,gate,selection,cfg,store,router,namespace)
        self.family = family

    def predict(self,observation):
        if self.family not in NEW_FAMILIES:
            prediction,features = super().predict(observation)
            prediction['effective_neighbors'] = float(prediction['retrieved_count'])
            return prediction,features
        if self.pending is not None:
            raise RuntimeError('Feedback or discard required before next prediction')
        outputs = self.experts.predict(SensorObservation(observation.readings))
        settings = self.selection['forgetting']
        retrieved,count,steps,effective = self.store.recent(self.namespace,observation,before_step=self.step,
            capacity=settings['capacity'],limit=settings['neighbors'],radius=settings['radius'],
            half_life=None if self.family=='recent_uniform' else settings['half_life'],
            permutation=self.shuffle if self.family=='shuffled_forgetting' else None)
        loss = self.history.fast.copy()
        if retrieved is not None:
            blend = settings['blend']*min(1.,count/self.cfg['evidence_warmup'])
            loss = (1-blend)*loss+blend*retrieved
        q = softmax(-self.cfg['routing_temperature']*loss)
        base = float(.05+.9*(q@outputs))
        features = self.history.gate_input(base)
        alpha = self.gate.alpha(features)
        self.pending = observation,outputs,base
        return {'probability':base,'gated_probability':float(.5+alpha*(base-.5)),
            'alpha':alpha,'route':q.tolist(),'retrieved_count':count,
            'retrieved_max_step':max(steps) if steps else None,'effective_neighbors':effective},features
