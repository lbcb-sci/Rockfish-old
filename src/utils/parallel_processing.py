from __future__ import annotations

from concurrent.futures import Future, Executor, ProcessPoolExecutor

from typing import *

T = TypeVar('T')


class SingleProcessExecutor(Executor):
    """ Mock executor used when number of workers is set to zero. """
    def __init__(self, initializer: Optional[Callable[..., None]]=None, initargs: Tuple[Any, ...]=()) -> None:
        super(SingleProcessExecutor, self).__init__()

        if initializer:
            initializer(*initargs)

    def __enter__(self: SingleProcessExecutor) -> SingleProcessExecutor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Optional[bool]:
        return

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        future = Future()
        if not future.set_running_or_notify_cancel():
            return future

        try:
            result = fn(*args, **kwargs)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            future.set_exception(e)
        else:
            future.set_result(result)

        return future


def get_executor(
        num_workers: int,
        initializer: Optional[Callable[..., None]]=None,
        initargs: Tuple[Any, ...]=()) -> Executor:
    """ Function that returns an executor according to the desired number of workers.

    :param num_workers: Number of workers
    :param initializer: Worker initializer function
    :param initargs: Worker initialization arguments
    :return: SingleProcessorExecutor if num_workers is 0, ProcessPoolExecutor if num_workers > 0, otherwise raise a
    ValueError exception
    """
    if num_workers == 0:
        return SingleProcessExecutor(initializer=initializer, initargs=initargs)
    elif num_workers > 0:
        return ProcessPoolExecutor(num_workers, initializer=initializer, initargs=initargs)
    else:
        raise ValueError(f'{num_workers} is not a valid argument for number of workers.')
