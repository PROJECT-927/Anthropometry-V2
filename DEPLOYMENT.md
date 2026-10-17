# Deploying Anthropometry V2

Get all updated V2 source files, including the new compute queue, memory helpers,
optimized pipeline, and templates. Updating only the Dockerfile will not update
the algorithms in the running image. Build from this directory:

```sh
docker build --pull -t anthropometry-v2:memory-fix .
```

Wait for active jobs to finish. If replacing the previous container named
`anthropometry-v2`, stop and remove that container after the build succeeds:

```sh
docker stop anthropometry-v2
docker rm anthropometry-v2
```

Start the rebuilt image with the same output volume:

```sh
docker run -d --name anthropometry-v2 --restart unless-stopped \
  -p 5000:5000 \
  -v anthropometry_v2_runs:/app/runs \
  anthropometry-v2:memory-fix
```

For a first deployment, skip the stop/remove commands. If the existing deployment
uses a different container name or volume, use those names instead. Removing the
container does not remove this named volume. Configure your instance's public
HTTPS URL/reverse proxy to forward to the app's port. The image defaults to port
5000; `PORT` can change the internal listening port.

The Dockerfile enables UTF-8 output, a headless plotting backend, native-library
thread limits, and an HTTP health check. Pipeline memory diagnostics are forwarded
to the container logs. The health check verifies the web server responds; it does
not execute a measurement or determine whether every input fits in memory.

```sh
docker inspect --format='{{.State.Health.Status}}' anthropometry-v2
docker logs --tail 100 anthropometry-v2
```

## Compute capacity

Keep **one Gunicorn worker and one container replica** with the current in-memory
job registry and compute queue. The Dockerfile fixes Gunicorn workers at one.
Gunicorn request threads handle uploads and polling; they do not each need a
pipeline process. Pipelines, viewer export, and STL export share a bounded queue.

| Environment variable | Default | Effect |
| --- | --- | --- |
| `MAX_CONCURRENT_COMPUTE_JOBS` | `1` | Heavy tasks running at once; additional users wait in the queue. |
| `MAX_PENDING_COMPUTE_JOBS` | `8` | Extra queued heavy tasks; a full queue returns HTTP 429. |
| `MAX_GRID_VOXELS` | `400000000` | Maximum cells per dense grid, checked before allocation. |
| `WEB_CONCURRENCY_THREADS` | `8` | Gunicorn request threads, independent of heavy task concurrency. |

Two uploads can be accepted while only one pipeline computes. The second page
shows that it is waiting and continues polling. STL requests use the same queue
so mesh export cannot bypass the memory budget. Increase compute concurrency only
after profiling representative jobs and leaving headroom for exports, the web
process, and the OS. Queue and live job state do not survive a server restart.
A persistent shared job system is required before adding workers or replicas.

The grid guard fails oversized jobs with a clear error. It does not change voxel
resolution or silently return measurements from a coarser reconstruction.

## Memory investigation

The earlier carving chunk patch bounded only temporary coordinate arrays.
Remaining large allocations included:

- `ndimage.sum` over all component labels, which implicitly required large
  integer and floating-point buffers across the whole volume.
- Per-row label images retained for every smoothing track.
- Earlier full hull copies kept alive by local variables in the pipeline.
- Padded integer distance fields during display shoulder-junction closing.
- Full coordinate arrays built before selecting viewer points; multiple float
  grids and retained temporary mesh arrays during STL export.
- Unlimited simultaneous pipeline/STL subprocesses.

The revised implementation uses two 2D projection lookups for carving, bounded
component histograms, row-local labels, prompt release of obsolete arrays,
equivalent boolean morphology, and smaller export intermediates. The browser
mask-only path also avoids importing the unused server MediaPipe inference runtime.

Local Windows measurements with the same Python environment and fixed 2 mm
resolution (fresh processes, OS-recorded peak resident memory):

| Workload | Previous carving chunks + old downstream code | Revised code |
| --- | ---: | ---: |
| 154,272,750-cell grid: carve, cleanup, smoothing, display carve/round | 3,196.5 MiB | 989.0 MiB |
| 267,025,500-cell grid: complete downstream pipeline with identical prepared masks/landmarks | 5,491.3 MiB | 1,639.0 MiB |

Both saved voxel arrays and final measurement JSON in the full comparison were
exactly equal. Two real mask-only Flask API submissions, including actual pipeline
subprocesses and viewer export, both completed; the second waited in the queue.
Their pipeline peaks were 1,648.9 and 1,650.4 MiB. These numbers are process RSS on
the local machine, not measurements of the deployed Linux container or a maximum
for every input. Docker was unavailable locally, so the container build and its
actual memory limit still need verification on the instance.

Dense 3D grids still have a memory cost: a 344-million-cell boolean grid alone
uses about 328 MiB; SciPy's full-volume int32 component labels use about 1.28 GiB.
The 15.9 GB observed remotely is not an inherent requirement of the measurement
formulas. Exact attribution of that remote peak requires the deployed logs,
grid shape, image revision, concurrency, and container limit.

## Inspect the deployed instance

Every pipeline logs source hashes, grid dimensions, current process RSS, and
cumulative process peak RSS at the expensive stages. On Linux cgroup v2 it also
records container usage/limit, peak (when available), and memory events. These
appear in container logs and each run's `pipeline.log`. Viewer/STL exports record
their own process peaks in `viewer.log` and `stl.log`.

```sh
docker logs -f anthropometry-v2
docker stats anthropometry-v2
docker inspect --format='MemoryLimitBytes={{.HostConfig.Memory}} OOMKilled={{.State.OOMKilled}}' anthropometry-v2
docker exec anthropometry-v2 sh -c 'cat /sys/fs/cgroup/memory.max; cat /sys/fs/cgroup/memory.events'
sudo journalctl -k --since '30 minutes ago' | grep -Ei 'oom|killed process|out of memory'
```

Container memory limits can be smaller than instance RAM. The container's
`OOMKilled` flag alone does not rule out a killed pipeline child. An incrementing
`oom_kill` event or a kernel OOM record can confirm the reason for SIGKILL.
Compare the printed source hashes with the image you intended to deploy.

To reproduce the carve/cleanup/display profile locally on an existing run:

```sh
python tools/benchmark_memory.py --fixture runs/EXISTING_RUN --round --output memory-result.json
python -m unittest discover -s tests -v
```

The fixture needs `front_normalized_mask.png`, `side_normalized_mask.png`, and
`visual_hull.npz`. Each revision should be measured in its own fresh process.
The benchmark does not require the original photographs or contact a server.

## Privacy preprocessing and torso-boundary errors

Browser preprocessing v2 restores operations present in the original local photo
path: bilinear probability resizing before thresholding, width-alignment padding,
largest-component cleanup, small enclosed-hole filling, and a second pose pass on
a crop around the subject with a 15% margin. It uses CPU/WASM inference and retains
the mobile image-size cap. The extra pose pass can take longer on a phone.
Browser and Python image resizing/inference are not guaranteed bit-for-bit equal.

Original photos and the crop stay in the browser. The upload contains the binary
silhouettes, mapped-back landmark coordinates, height and a preprocessing version.
Older clients receive the binary cleanup on the server; new clients mark their
masks as preprocessed so the hole-filling step is not applied twice.

Torso tracing previously ignored its supplied centreline when selecting a run;
the widest detached arm could seed the trace even when a central torso was visible.
The corrected selection requires a torso interval straddling the centreline.
Unsupported ends are trimmed, short internal gaps up to approximately 5 mm can be
interpolated, and larger gaps still fail instead of producing invented widths.
Measurement errors now identify the front/side view where available.

Deploy the updated static asset, template, Python code and Dockerfile together.
After rebuilding/replacing the container, reload the page and reselect the photos.
`pipeline.log` records the preprocessing version and torso/measurement module
hashes. If the error persists, investigate that run's `front_normalized_mask.png`,
`side_normalized_mask.png`, `front_landmarks.json`, `side_landmarks.json` and log.
Those artifacts are sufficient to investigate the server boundary failure without
transferring the original photographs. They remain sensitive body-derived data.

## Compare two app versions on Windows

`tools/monitor_memory.py` runs independently of either app's diagnostic code. It
starts the selected project and samples the web server and all discovered child
processes every 100 ms, including pipeline, viewer, and STL processes. Its RAM
figure excludes the browser and unrelated applications. Summed process RSS can
count shared pages more than once; use the same monitor for both versions.
Sampled peaks can miss short spikes. Windows per-process high-water marks are
also saved in the JSON, but are not summed as if they occurred simultaneously.

From V2, install the optional dependency once (not part of production requirements):

```powershell
.\.test_venv\Scripts\python.exe -m pip install -r requirements-monitoring.txt
```

Stop existing local servers on port 5000. Use the same Python interpreter for both
runs to hold dependency versions constant. Replace the old project path below if
your unmodified version is stored elsewhere:

```powershell
.\.test_venv\Scripts\python.exe tools\monitor_memory.py --project 'C:\Users\SUMANTH S SHETTY\Documents\Anthropometry - Copy' --python '.\.test_venv\Scripts\python.exe' --label before
```

Open `http://127.0.0.1:5000/`, select the two photos, and enter your height. Wait for
the measurements and viewer to finish, then return to PowerShell and press Enter.
The monitor stops that server and saves timestamped CSV, JSON, and server logs in
`memory_profiles/`. It also writes `before.latest.json` for easy comparison.

Run the modified version using the same photos, height, and voxel settings:

```powershell
.\.test_venv\Scripts\python.exe tools\monitor_memory.py --project . --python '.\.test_venv\Scripts\python.exe' --label after
```

Again, wait for the viewer and press Enter in PowerShell. Compare the saved peaks:

```powershell
.\.test_venv\Scripts\python.exe tools\monitor_memory.py --compare memory_profiles\before.latest.json memory_profiles\after.latest.json
```

Repeat three times and keep the timestamped results. First measure one user with
no STL request; then repeat with an STL request in both versions, and finally with
two users. Keep the laptop's power mode and other workload consistent. Compare
the returned measurements as well. The displayed pipeline time is observed child
process lifetime, not total browser-to-result time; native memory figures on the
deployed Linux instance can differ from these local Windows figures.
