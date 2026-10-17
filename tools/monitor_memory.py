"""Start either app version and record RAM for its entire process tree.

Use the same monitor for both versions. Press Enter after measurements, viewer,
and any requested STL have finished; the monitor saves results and stops its app.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import os


MIB = 2**20


def find_python(project, explicit):
    if explicit:
        executable = Path(explicit).expanduser().resolve()
        if not executable.is_file():
            raise ValueError(f"Python interpreter not found: {executable}")
        return executable
    for name in (".test_venv", ".venv", "venv"):
        for relative in ("Scripts/python.exe", "bin/python"):
            candidate = project / name / relative
            if candidate.is_file():
                return candidate
    return Path(sys.executable)


def compare(paths):
    before, after = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    print(f"{'Run':20} {'Peak tree RAM (MiB)':>21} {'First pipeline (s)':>21}")
    for result in (before, after):
        duration = result.get("first_pipeline_observed_seconds")
        duration = f"{duration:.2f}" if duration is not None else "not observed"
        print(f"{result['label']:20} {result['peak_tree_rss_mib']:21.1f} {duration:>21}")
    if before["peak_tree_rss_mib"]:
        reduction = 100 * (1 - after["peak_tree_rss_mib"] / before["peak_tree_rss_mib"])
        print(f"Peak RAM reduction: {reduction:.1f}%")
    print("Pipeline time is sampled process lifetime; compare matching inputs and stages.")


def monitor(args, psutil):
    project = args.project.expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"Project directory not found: {project}")
    python = find_python(project, args.python)
    script = project / args.script
    if not script.is_file():
        raise ValueError(f"App script not found: {script}")
    if args.interval < 0.05:
        raise ValueError("Use a sampling interval of at least 0.05 seconds.")
    with socket.socket() as probe:
        probe.settimeout(0.3)
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise ValueError(f"Port {args.port} is already in use. Stop the other server first.")

    folder = args.output.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", args.label).strip("_") or "run"
    stem = f"{label}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    csv_path, json_path, log_path = [folder / (stem + suffix) for suffix in (".csv", ".json", ".server.log")]
    env = dict(os.environ, PORT=str(args.port), HOST="0.0.0.0",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    done = threading.Event()
    records = {}
    samples = 0
    peak = 0
    idle_before_first_pipeline = None
    previous_idle = None
    started = time.monotonic()
    next_display = 0
    last_elapsed = 0
    server_exit = None

    with log_path.open("w", encoding="utf-8") as log, csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        process = subprocess.Popen([str(python), "-u", str(script)], cwd=project, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        known = {(root.pid, root.create_time()): root}
        writer = csv.DictWriter(csv_file, fieldnames=("elapsed_seconds", "tree_rss_mib", "server_rss_mib",
                                                     "children_rss_mib", "process_count", "phase"))
        writer.writeheader()
        print(f"Started {project}\nOpen http://127.0.0.1:{args.port}/ and run your measurement.", flush=True)
        print("Wait for the results and viewer. Press Enter here to save and stop this server.", flush=True)
        print(f"Server output: {log_path}", flush=True)

        def read_enter():
            try:
                input()
            except EOFError:
                return
            done.set()

        threading.Thread(target=read_enter, daemon=True).start()
        try:
            while not done.is_set():
                elapsed = time.monotonic() - started
                last_elapsed = elapsed
                try:
                    for child in root.children(recursive=True):
                        known[(child.pid, child.create_time())] = child
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                total = server_rss = 0
                count = 0
                active_roles = set()
                for key, item in list(known.items()):
                    try:
                        if not item.is_running():
                            if key in records and records[key]["ended_seconds"] is None:
                                records[key]["ended_seconds"] = elapsed
                            continue
                        with item.oneshot():
                            memory = item.memory_info()
                            command = item.cmdline()
                        role = next((name for name in ("pipeline", "export_hull_viewer", "export_stl")
                                     if any(Path(part).name == name + ".py" for part in command)), "server")
                        record = records.setdefault(key, {"pid": item.pid, "role": role,
                                                          "first_seen_seconds": round(elapsed, 3),
                                                          "ended_seconds": None, "peak_rss_bytes": 0})
                        # Windows exposes the OS high-water mark as well as sampled RSS.
                        record["peak_rss_bytes"] = max(record["peak_rss_bytes"], memory.rss,
                                                       getattr(memory, "peak_wset", 0))
                        total += memory.rss
                        count += 1
                        if item.pid == root.pid:
                            server_rss = memory.rss
                        active_roles.add(role)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        if key in records and records[key]["ended_seconds"] is None:
                            records[key]["ended_seconds"] = elapsed
                peak = max(peak, total)
                heavy = active_roles - {"server"}
                if "pipeline" in heavy and idle_before_first_pipeline is None:
                    idle_before_first_pipeline = previous_idle
                if not heavy:
                    previous_idle = total
                writer.writerow({"elapsed_seconds": round(elapsed, 3), "tree_rss_mib": round(total/MIB, 3),
                                 "server_rss_mib": round(server_rss/MIB, 3),
                                 "children_rss_mib": round((total-server_rss)/MIB, 3),
                                 "process_count": count, "phase": "+".join(sorted(heavy)) or "idle"})
                csv_file.flush()
                samples += 1
                if elapsed >= next_display:
                    print(f"Current: {total/MIB:.1f} MiB | Peak: {peak/MIB:.1f} MiB | "
                          f"Phase: {'+'.join(sorted(heavy)) or 'idle'}", flush=True)
                    next_display = elapsed + 5
                if process.poll() is not None:
                    server_exit = process.returncode
                    print(f"Server exited with code {server_exit}; inspect {log_path}", flush=True)
                    break
                if args.duration and elapsed >= args.duration:
                    break
                done.wait(args.interval)
        except KeyboardInterrupt:
            print("\nSaving memory results…", flush=True)
        finally:
            # Only stop the process tree launched by this monitor. Keep other
            # Python sessions and browser processes running on the computer.
            try:
                for child in root.children(recursive=True):
                    known[(child.pid, child.create_time())] = child
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            running = []
            for item in reversed(list(known.values())):
                try:
                    if item.is_running():
                        item.terminate()
                        running.append(item)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            _, alive = psutil.wait_procs(running, timeout=5)
            for item in alive:
                try:
                    item.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            process.wait(timeout=10)

    pipelines = sorted((r for r in records.values() if r["role"] == "pipeline"),
                       key=lambda r: r["first_seen_seconds"])
    first_time = None
    if pipelines and pipelines[0]["ended_seconds"] is not None:
        first_time = round(pipelines[0]["ended_seconds"] - pipelines[0]["first_seen_seconds"], 3)
    result = {"label": args.label, "project": str(project), "python": str(python),
              "sample_interval_seconds": args.interval, "samples": samples,
              "monitor_duration_seconds": round(last_elapsed, 3), "peak_tree_rss_mib": round(peak/MIB, 3),
              "idle_before_first_pipeline_mib": round(idle_before_first_pipeline/MIB, 3)
                  if idle_before_first_pipeline is not None else None,
              "first_pipeline_observed_seconds": first_time,
              "server_exit_before_stop": server_exit, "processes": list(records.values()),
              "csv": str(csv_path), "server_log": str(log_path),
              "notes": "Tree RSS is sampled and sums resident memory of all processes; shared pages may be counted more than once. Browser and OS memory are excluded."}
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    latest_path = folder / (label + ".latest.json")
    latest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nPeak server + child-process RAM: {peak/MIB:.1f} MiB ({peak/2**30:.2f} GiB)", flush=True)
    print(f"Summary: {json_path}\nCSV: {csv_path}", flush=True)
    print(f"Latest summary for comparison: {latest_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, help="Folder containing the version to run")
    parser.add_argument("--label", default="run", help="For example before or after")
    parser.add_argument("--python", help="Optional explicit Python interpreter for that project")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"memory_profiles")
    parser.add_argument("--script", default="app.py")
    parser.add_argument("--duration", type=float, help="Automatically stop after this many seconds")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE_JSON", "AFTER_JSON"))
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
        return
    if not args.project:
        parser.error("Supply --project, or --compare with two result files.")
    try:
        import psutil
    except ImportError:
        parser.exit(1, "Install the monitoring dependency first: python -m pip install -r requirements-monitoring.txt\n")
    try:
        monitor(args, psutil)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
