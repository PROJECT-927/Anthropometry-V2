"""Run the production browser photo preprocessor and retain its exact derived inputs.

The temporary server binds only loopback. Source photos stay local and only
derived masks/landmarks are written, exactly as in the web application's form.
"""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/rectification/browser_inputs")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT/"runs/rectification/reported_before.json").read_text(encoding="utf-8"))
    subjects = [s for s in prior["subjects"] if s["name"].casefold() not in {"sohan", "tanmay", "shashank"}]
    jobs = [{"name":s["name"].casefold(), "photos":s["photos"]} for s in subjects]
    photos = {filename for s in subjects for filename in s["photos"].values()}
    allowed = {s["name"].casefold() for s in subjects}
    from app import app
    with app.test_client() as client:
        page = client.get("/").get_data(as_text=True)
    script = """<script>
window.addEventListener('load', async () => {
 try {
  for (const job of JOBS) {
   const files = {};
   for (const view of ['front','side']) {
    const response = await fetch('/audit-photo/'+job.photos[view]);
    if (!response.ok) throw new Error('Photo read failed');
    const blob = await response.blob();
    files[view] = new File([blob],job.photos[view],{type:blob.type});
    document.getElementById('prev-'+view).src = URL.createObjectURL(blob);
   }
   const inputs = await window.prepareAnthropometryInputs(files.front,files.side,()=>{});
   for (const view of ['front','side']) {
    await fetch('/audit-save/'+job.name+'/'+view+'_mask.png',{method:'POST',body:inputs[view].silhouette});
    await fetch('/audit-save/'+job.name+'/'+view+'_landmarks.json',{method:'POST',body:inputs[view].landmarks});
   }
   const manifest={preprocessing:inputs.preprocessing,runtime:MEDIAPIPE_VERSION,model:MODEL_URL,
                   frontRefinement:inputs.front.refinement,sideRefinement:inputs.side.refinement};
   await fetch('/audit-save/'+job.name+'/preprocessing.json',{method:'POST',body:JSON.stringify(manifest)});
   await fetch('/audit-progress',{method:'POST',body:job.name});
  }
  await fetch('/audit-done',{method:'POST',body:'complete'});
 } catch(error) {
  await fetch('/audit-done',{method:'POST',body:'ERROR: '+(error.stack || error.message)});
 }
});
</script>""".replace("JOBS", json.dumps(jobs))
    page = page.replace("</body>", script+"</body>")
    finished = threading.Event()
    outcome = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                data, mime = page.encode(), "text/html; charset=utf-8"
            elif path.startswith("/audit-photo/") and path.rsplit("/",1)[1] in photos:
                filename = path.rsplit("/",1)[1]
                data = (ROOT/"photoset"/filename).read_bytes()
                mime = "image/png" if filename.endswith(".png") else "image/jpeg"
                if filename == "lakyside.jpeg":
                    import cv2
                    image = cv2.imread(str(ROOT/"photoset"/filename))[128:1216,240:720]
                    data, mime = cv2.imencode(".png", image)[1].tobytes(), "image/png"
            else:
                with app.test_client() as client:
                    response = client.get(path)
                    data, mime = response.data, response.content_type
                    if response.status_code != 200:
                        self.send_error(response.status_code)
                        return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            parts = self.path.strip("/").split("/")
            if len(parts)==3 and parts[0]=="audit-save" and parts[1] in allowed and parts[2] in {
                "front_mask.png", "side_mask.png", "front_landmarks.json", "side_landmarks.json", "preprocessing.json"}:
                target = output/parts[1]
                target.mkdir(exist_ok=True)
                (target/parts[2]).write_bytes(data)
            elif self.path=="/audit-progress":
                print(data.decode()+": production browser inputs captured", flush=True)
            elif self.path=="/audit-done":
                outcome.append(data.decode())
                finished.set()
            else:
                self.send_error(400)
                return
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1",0), Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    browser = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    profile = output/"browser_profile"
    log = (output/"browser.log").open("w",encoding="utf-8")
    process = subprocess.Popen([str(browser), "--headless=new", "--disable-gpu", "--disable-breakpad", "--no-first-run",
                                "--no-default-browser-check", "--user-data-dir="+str(profile),
                                f"http://127.0.0.1:{server.server_port}/"],stdout=log,stderr=subprocess.STDOUT)
    try:
        start = time.monotonic()
        while not finished.wait(1):
            if process.poll() is not None:
                raise RuntimeError("Browser exited; inspect "+str(output/"browser.log"))
            if time.monotonic()-start > 240:
                raise RuntimeError("Browser preprocessing timed out; inspect "+str(output/"browser.log"))
        if outcome[0] != "complete":
            raise RuntimeError(outcome[0])
        print("All 8 subjects captured with the production browser preprocessor.",flush=True)
    finally:
        process.terminate()
        process.wait(timeout=20)
        server.shutdown()
        log.close()


if __name__ == "__main__":
    main()
