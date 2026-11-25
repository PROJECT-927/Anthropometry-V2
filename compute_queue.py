"""First-come-first-served slots for heavy subprocess work in this one web worker."""

import os
import threading


def available_cpus():
    """CPUs this process may run on (respects taskset/cpuset, unlike os.cpu_count)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on Windows/macOS
        return os.cpu_count() or 1


class ComputeQueue:
    """
    At most `workers` jobs hold a slot at once, and at most `pending` more wait.

    A "heavy" job (STL export, several times the memory of a pipeline run) also
    never runs beside another heavy one; lighter jobs behind it are not held up.
    workers <= 0 removes the limit (heavy jobs still go one at a time).
    """

    def __init__(self, workers, pending):
        self.workers, self.pending = workers, pending
        self._cond = threading.Condition()
        self._waiting = []  # tickets, oldest first
        self._running = 0
        self._heavy_running = False

    def join(self, key, heavy=False):
        """Takes a place in line without blocking; None when the line is full."""
        ticket = (object(), key, heavy)
        with self._cond:
            free = max(self.workers - self._running, 0)
            if self.workers > 0 and len(self._waiting) >= self.pending + free:
                return None
            self._waiting.append(ticket)
        return ticket

    def _next(self):
        if self.workers > 0 and self._running >= self.workers:
            return None
        return next((t for t in self._waiting if not (t[2] and self._heavy_running)), None)

    def acquire(self, ticket, cancelled):
        """Blocks until it is this ticket's turn; False if cancelled() first."""
        with self._cond:
            while True:
                if cancelled():
                    self._waiting.remove(ticket)
                    self._cond.notify_all()
                    return False
                if self._next() is ticket:
                    self._waiting.remove(ticket)
                    self._running += 1
                    self._heavy_running |= ticket[2]
                    self._cond.notify_all()
                    return True
                self._cond.wait()

    def release(self, ticket):
        with self._cond:
            self._running -= 1
            if ticket[2]:
                self._heavy_running = False
            self._cond.notify_all()

    def wake(self):
        """Makes waiting jobs re-check cancelled(), e.g. after a deletion."""
        with self._cond:
            self._cond.notify_all()

    def ahead(self, key):
        """How many jobs are in line before this one."""
        with self._cond:
            return next((i for i, t in enumerate(self._waiting) if t[1] == key), 0)
