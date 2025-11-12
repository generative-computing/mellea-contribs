
import functools
from mellea.helpers.event_loop_helper import _run_async_in_thread

def sync_wrapper(async_fn):
    """Wrap an async function so it can be called synchronously."""
    @functools.wraps(async_fn)
    def wrapper(*args, **kwargs):
        return _run_async_in_thread(async_fn(*args, **kwargs))
    return wrapper


