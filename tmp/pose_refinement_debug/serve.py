from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import subprocess,re,json
ROOT=Path.cwd()
PHOTO=Path(r'C:/Users/SUMANTH S SHETTY/Documents/Anthropometry/photoset/thilakside.jpeg')
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split('?')[0]
        known={'/fixture.jpeg':(PHOTO,'image/jpeg'),'/models/pose_landmarker_heavy.task':(ROOT/'pose_landmarker_heavy.task','application/octet-stream'),'/static/anthropometry-preprocessing.js':(ROOT/'static/anthropometry-preprocessing.js','text/javascript')}
        if path in known:
            f,kind=known[path]; data=f.read_bytes()
        elif path in ['/head','/current','/previous']:
            if path=='/current': html=(ROOT/'templates/index.html').read_text(encoding='utf-8')
            else:html=subprocess.check_output(['git','show',('HEAD^' if path=='/previous' else 'HEAD')+':templates/index.html'],cwd=ROOT).decode('utf-8')
            script=next(s for _,s in re.findall(r'<script([^>]*)>([\s\S]*?)</script>',html) if 'const MEDIAPIPE_VERSION' in s)
            script=script.replace('const MEDIAPIPE_VERSION', 'const MEDIAPIPE_VERSION',1)
            harness=r'''
const recorded=[],messages=[];
for (const name of ['warn','error']) {
 const original=console[name].bind(console);
 console[name]=(...args)=>{messages.push({level:name,args:args.map(a=>a?.stack||String(a))});original(...args);};
}
const realGetLandmarker=getLandmarker;
getLandmarker=async()=>{
 const detector=await realGetLandmarker();
 return {detect(canvas){
   const entry={width:canvas.width,height:canvas.height};recorded.push(entry);
   try {const result=detector.detect(canvas);entry.poseCount=result.landmarks?.length||0;return result;}
   catch(error){entry.error=error?.stack||String(error);throw error;}
 }};
};
const begun=performance.now();
preparePhoto({type:'image/jpeg',name:'thilakside.jpeg'},'side').then(result=>{
 document.getElementById('report').textContent=JSON.stringify({status:'success',refinement:result.refinement||'original_single_pass',elapsedMs:performance.now()-begun,recorded,messages,landmarks:JSON.parse(result.landmarks)},null,2);
}).catch(error=>{
 document.getElementById('report').textContent=JSON.stringify({status:'failed',error:error?.stack||String(error),elapsedMs:performance.now()-begun,recorded,messages},null,2);
});
'''
            data=('<!doctype html><title>Local pose regression '+path+'</title><h1>Local pose regression '+path+'</h1><p>The supplied photo is processed locally. Waiting for browser model...</p><pre id="report">Loading</pre><img id="prev-side" src="/fixture.jpeg" style="max-height:320px"><script src="/static/anthropometry-preprocessing.js"></script><script>'+script+'\n'+harness+'</script>').encode('utf-8');kind='text/html;charset=utf-8'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def log_message(self,fmt,*args):print(fmt%args,flush=True)
print('Pose reproduction at http://127.0.0.1:8765/head (original memory commit) and /current (working tree).',flush=True)
ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
