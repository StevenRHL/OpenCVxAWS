"""Only locally trained, checksum-verified model artifacts are loaded.

How a model is fed and read comes from its card, not from its name: `input` says whether
it takes a per-frame vector or a windowed descriptor, and `scoring` says whether a higher
value means probability or anomaly. Hardcoding either lets a retrained model of a
different family fail silently at runtime.
"""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FRAME_VECTOR='frame_vector'
CLIP_DESCRIPTOR='clip_descriptor'

class Models:
    def __init__(self):
        self.loaded={};self.status={}
        for name in ['fall','activity']:
            meta=ROOT/'models'/f'{name}.json'
            if not meta.exists():self.status[name]='Not trained; rule prototype only' if name=='fall' else 'Not trained; no activity predictions';continue
            try:
                card=json.loads(meta.read_text());model_path=ROOT/'models'/card['artifact']
                if model_path.parent!=ROOT/'models':raise ValueError('Invalid model path')
                if hashlib.sha256(model_path.read_bytes()).hexdigest()!=card['sha256']:raise ValueError('Model checksum mismatch')
                import joblib
                model=joblib.load(model_path)
                expected=int(card.get('n_features',getattr(model,'n_features_in_',0) or 0))
                actual=int(getattr(model,'n_features_in_',expected) or expected)
                if expected and actual and expected!=actual:
                    raise ValueError(f'Card expects {expected} features, artifact takes {actual}')
                if card.get('scoring')=='anomaly_score' and not hasattr(model,'score_samples'):
                    raise ValueError('Card declares anomaly scoring but the artifact cannot produce it')
                if card.get('scoring','probability')=='probability' and not hasattr(model,'predict_proba'):
                    raise ValueError('Card declares probability scoring but the artifact cannot produce it')
                self.loaded[name]=(model,card)
                self.status[name]=card.get('status','Experimental local model')
            except Exception as exc:self.status[name]=f'Unavailable: {exc}'

    def disclosures(self):
        """Snapshot only cards whose artifacts passed the actual inference loader."""
        fields = ('run_id', 'sha256', 'alert_caveat', 'release_cleared', 'release_status',
                  'disclosure_evidence')
        return {name: {key: card[key] for key in fields if key in card}
                for name, (_, card) in self.loaded.items()}

    def input_kind(self,name):
        if name not in self.loaded:return None
        return self.loaded[name][1].get('input',FRAME_VECTOR)

    def score(self,name,vector,mask=None):
        if name not in self.loaded:return None
        model,card=self.loaded[name]
        x=np.asarray(vector,dtype=float).reshape(1,-1)
        if card.get('input',FRAME_VECTOR)==FRAME_VECTOR and card.get('with_mask'):
            if mask is None:return None
            x=np.concatenate([x,np.asarray(mask,dtype=float).reshape(1,-1)],axis=1)
        if not np.isfinite(x).all():return None
        expected=int(card.get('n_features',x.shape[1]))
        if x.shape[1]!=expected:return None
        if card.get('scoring')=='anomaly_score':value=float(-model.score_samples(x)[0])
        else:value=float(model.predict_proba(x)[0,1])
        return {'score':value,'threshold':card['threshold'],'positive':value>=card['threshold'],
                'version':card['run_id'],'score_type':card.get('scoring','probability')}
