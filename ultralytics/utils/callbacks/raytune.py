# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    try:
        if ray.train._internal.session.get_session():  # fixed deprecated function call
            metrics = trainer.metrics
            metrics["epoch"] = trainer.epoch
            session.report(metrics)
    except AttributeError:
        # Fallback for different Ray versions
        try:
            if ray.tune.is_session_enabled():
                metrics = trainer.metrics
                metrics["epoch"] = trainer.epoch
                session.report(metrics)
        except (AttributeError, NameError):
            pass  # Ray Tune not available or session not enabled


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
