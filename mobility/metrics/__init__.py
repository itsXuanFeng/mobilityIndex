"""
Metric registry. A metric is a function of mobility.states.States -> float, registered with @metric. It is computed
by the same rule from the radar states and the MoCap states, so the two can be compared per recording.

    from mobility.metrics import metric

    @metric('my_metric', category=2, unit='count')
    def my_metric(s):
        return float(...)

Put it in the module of its category (category1.py, category2.py, category3.py); it is then available to
compute_metrics and to the evaluation scripts.
"""
from dataclasses import dataclass


@dataclass
class Metric:
    name: str
    category: int
    unit: str
    fn: callable
    doc: str


METRICS = {}


def metric(name, category, unit):
    def register(fn):
        METRICS[name] = Metric(name, category, unit, fn, (fn.__doc__ or '').strip())
        return fn
    return register


def compute_metrics(states, category=None):
    """{metric name: value} for all registered metrics (of one category)"""
    return {m.name: m.fn(states) for m in METRICS.values() if category is None or m.category == category}


from . import category1, category2, category3  # noqa: E402,F401  (registers the metrics)
