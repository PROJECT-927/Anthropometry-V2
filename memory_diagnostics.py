"""Small, dependency-free process memory snapshots for deployment logs."""

import ctypes
import json
import os
import sys
import time


def process_memory():
    """Return resident and OS-recorded peak resident memory in bytes."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)

    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    current = None
    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", encoding="ascii") as f:
            current = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return current, int(peak)


def memory_snapshot(stage):
    try:
        current, peak = process_memory()
        data = {"stage": stage, "pid": os.getpid(), "peak_rss_mib": round(peak / 2**20, 1)}
        if current is not None:
            data["rss_mib"] = round(current / 2**20, 1)
        # cgroup v2 metrics describe the whole container, not only this process.
        for name in ("memory.current", "memory.max", "memory.peak", "memory.events"):
            path = os.path.join("/sys/fs/cgroup", name)
            if os.path.isfile(path):
                with open(path, encoding="ascii") as f:
                    data["cgroup_" + name] = f.read().strip()
        print("[memory] " + json.dumps(data, sort_keys=True), flush=True)
        return data
    except (OSError, ValueError):
        return {}


def tracked_call(stage, function, *args, **kwargs):
    memory_snapshot(stage + ":start")
    started = time.monotonic()
    try:
        return function(*args, **kwargs)
    finally:
        memory_snapshot(stage + ":end")
        print(f"[timing] {stage}: {time.monotonic() - started:.2f}s", flush=True)
