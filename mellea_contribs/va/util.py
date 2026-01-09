
import functools
from mellea.helpers.event_loop_helper import _run_async_in_thread


def session_wrapper(fn):
    """Wrap an async function so it can be called synchronously."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return fn(self.backend, self.ctx, *args, **kwargs)
    return wrapper



def sync_wrapper(async_fn):
    """Wrap an async function so it can be called synchronously."""
    @functools.wraps(async_fn)
    def wrapper(*args, **kwargs):
        return _run_async_in_thread(async_fn(*args, **kwargs))
    return wrapper


