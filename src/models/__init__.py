"""MODELS registry: name -> build(**params) returning an sklearn-style estimator.

Register new models here; evaluate.py and predict.py look them up by name.
"""

from src.models.baselines import PriorBaseline, TransitionBaseline

MODELS = {
    "prior": PriorBaseline,
    "transition": TransitionBaseline,
}


def build(name: str, **params):
    """Instantiate a registered model by name."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**params)
