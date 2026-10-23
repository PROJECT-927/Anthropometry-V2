"""Controlled failure-path probes, distinct from real-image validation."""
from pathlib import Path
import sys,json,tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'integrated'))
import run_pipeline as p
import mediapipe_silhouette as segmentation
import pose_normalization as pose
from voxel_memory import check_grid_size,GridMemoryError
records=[]
def rejects(name,fn,reason):
    try:fn()
    except Exception as e:
        assert reason in str(e),(name,str(e));records.append({'case':name,'expected':reason,'observed':str(e),'pass':True});return
    raise AssertionError(name+' not rejected')
probe_dir=ROOT/'review/control_inputs';probe_dir.mkdir(exist_ok=True)
with nullcontext(probe_dir) as d:
    image=np.zeros((200,200,3),np.uint8);file=Path(d)/'image.png';assert cv2.imwrite(str(file),image)
    class Detector:
        def __init__(self,result):self.result=result
        def detect(self,image):return self.result
    class Mask:
        def __init__(self,a):self.a=a
        def numpy_view(self):return self.a
    with patch.object(segmentation,'_detector',return_value=Detector(SimpleNamespace(pose_landmarks=[],segmentation_masks=[]))):
        rejects('missing_pose',lambda:p.raw_view(file,'unused'),'missing_pose')
    with patch.object(segmentation,'_detector',return_value=Detector(SimpleNamespace(pose_landmarks=[1],segmentation_masks=[]))):
        rejects('missing_segmentation',lambda:p.raw_view(file,'unused'),'missing_segmentation')
    fg=np.zeros((200,200),np.float32);fg[20:180,70:130]=1
    result=SimpleNamespace(pose_landmarks=[1],segmentation_masks=[Mask(fg)])
    with patch.object(segmentation,'_detector',return_value=Detector(result)),patch.object(pose,'normalize_pose',return_value=(fg,{'R':np.eye(2),'s':1.,'t':np.array([500.,0.])})):
        rejects('canvas_clipping',lambda:p.raw_view(file,'unused'),'foreground_lost_by_alignment_canvas')
    fg[:21,70:130]=1
    with patch.object(segmentation,'_detector',return_value=Detector(result)):
        rejects('input_border',lambda:p.raw_view(file,'unused'),'input_mask_border_contact')
    mask=np.zeros((100,100),np.uint8);mask[10:50,10:50]=1;mask[60:90,60:90]=1
    rejects('material_fragment',lambda:p.dominant_mask(mask),'ambiguous_foreground_components_material_fraction')
    rejects('empty',lambda:p.dominant_mask(np.zeros((20,20),np.uint8)),'empty_foreground')
    rejects('grid_budget',lambda:check_grid_size((401,1000,1000)),'limit')
    args=SimpleNamespace(output=str(Path(d)/'badstature'),height_cm=float('nan'),attempt_id='control',evidence_class='controlled_failure_probe',mode='silhouettes',front='missing',side='missing')
    record=p.execute(args);assert record['reason']=='invalid_stature';records.append({'case':'invalid_stature_attempt_record','pass':True})
(ROOT/'review/integrated_control_results.json').write_text(json.dumps({'scope':'mocked perception control paths, not real capture success or accuracy','cases':records},indent=2)+'\n')
print('PASS',len(records),'controlled probes')
