"""Bound heavy subprocess work within this single web worker."""

from concurrent.futures import ThreadPoolExecutor
import logging
import threading


class ComputeQueue:
    def __init__(self, workers=1, pending=8):
        if workers < 1 or pending < 0:
            raise ValueError("Compute workers must be >= 1 and pending capacity >= 0.")
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="compute")
        self.capacity = threading.BoundedSemaphore(workers + pending)

    def submit(self, function, *args, on_error=None):
        if not self.capacity.acquire(blocking=False):
            return False

        def run():
            try:
                function(*args)
            except Exception as error:
                logging.getLogger(__name__).exception("Heavy compute job failed")
                if on_error is not None:
                    on_error(error)
            finally:
                self.capacity.release()

        try:
            self.executor.submit(run)
        except Exception:
            self.capacity.release()
            raise
        return True
