"""Regression checks for torso selection and the photo-free preprocessing path."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from measurement_extraction import _torso_bounds_at, MeasurementError
from mediapipe_silhouette import keep_largest_component, fill_small_holes
from pose_normalization import _bbox_from_mask


class TorsoTraceTests(unittest.TestCase):
    def test_detached_wide_arm_cannot_seed_torso_trace(self):
        mask = np.zeros((80, 800), dtype=np.uint8)
        mask[:, 350:451] = 1
        mask[:, 10:181] = 1  # Wider than the torso, and present throughout band.
        bounds = _torso_bounds_at(mask, 40, 0, 1., 400., band_mm=30)
        np.testing.assert_allclose(bounds, (-50, 50), atol=1e-10)

    def test_arm_cannot_capture_trace_after_a_brief_join(self):
        mask = np.zeros((100, 500), dtype=np.uint8)
        mask[:, 200:301] = 1
        mask[:15, 50:301] = 1
        mask[15:, 50:221] = 1
        # A background gap after the first rows separates arm and torso.
        mask[15:, 180:200] = 0
        lo, hi = _torso_bounds_at(mask, 50, 0, 1., 250., band_mm=30)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_one_sided_mask_still_fails_with_view_context(self):
        mask = np.zeros((80, 800), dtype=np.uint8)
        mask[:, 10:181] = 1
        with self.assertRaisesRegex(MeasurementError, "side view.*centreline"):
            _torso_bounds_at(mask, 40, 0, 1., 400., band_mm=30, view_name="side")

    def test_large_midline_gap_is_not_fabricated(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[:, 60:141] = 1
        mask[40:60, 90:111] = 0
        with self.assertRaisesRegex(MeasurementError, "gap.*interpolate"):
            _torso_bounds_at(mask, 50, 0, 1., 100., band_mm=35)

    def test_short_midline_gap_can_be_interpolated(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[:, 60:141] = 1
        mask[49:51, 90:111] = 0
        np.testing.assert_allclose(_torso_bounds_at(mask, 50, 0, 1., 100., band_mm=30),
                                   (-40, 40), atol=1e-10)

    def test_preprocessed_masks_are_not_cleaned_twice(self):
        import pipeline
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            mask_path, landmarks_path = folder/'mask.png', folder/'landmarks.json'
            cv2.imwrite(str(mask_path), np.ones((20, 20), np.uint8)*255)
            landmarks_path.write_text('{}')
            with patch.object(pipeline, 'fill_small_holes', side_effect=lambda m: m) as clean, \
                    patch.object(pipeline, 'keep_largest_component', side_effect=lambda m: m), \
                    patch.object(pipeline, 'normalize_pose_from_landmarks', return_value=(None, {})):
                pipeline.process_view(None, 'front', 173, mask_path=mask_path,
                                      landmarks_path=landmarks_path, masks_preprocessed=True)
                clean.assert_not_called()
                pipeline.process_view(None, 'front', 173, mask_path=mask_path,
                                      landmarks_path=landmarks_path, masks_preprocessed=False)
                clean.assert_called_once()


@unittest.skipUnless(shutil.which('node'), 'Node.js is required for browser helper equivalence checks')
class BrowserPreprocessingTests(unittest.TestCase):
    def node(self, program, data):
        result = subprocess.run(['node', '-e', program], input=json.dumps(data), text=True,
                                cwd=ROOT, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_browser_binary_cleanup_and_crop_match_original_python(self):
        masks = []
        # Central body, detached speck, small enclosed hole, large leg gap.
        mask = np.zeros((91, 71), np.uint8)
        mask[10:81, 20:51] = 1
        mask[20:23, 30:32] = 0
        mask[60:81, 33:37] = 0
        mask[2:5, 2:5] = 1
        masks.append(mask)
        rng = np.random.default_rng(106)
        for _ in range(8):
            image = np.zeros((51, 43), np.uint8)
            image[5:46, 8:35] = 1
            image[rng.random(image.shape)<0.04] = 0
            masks.append(image)
        data = [{'width':m.shape[1], 'height':m.shape[0], 'mask':m.ravel().tolist()} for m in masks]
        result = self.node("""
            require('./static/anthropometry-preprocessing.js');
            const fs=require('fs'), cases=JSON.parse(fs.readFileSync(0,'utf8'));
            console.log(JSON.stringify(cases.map(c=>{
              const r=AnthropometryPreprocessing.cleanSilhouette(c.mask,c.width,c.height);
              return {mask:Array.from(r.mask), crop:r.crop};
            })));
        """, data)
        for mask, actual in zip(masks, result):
            expected = fill_small_holes(keep_largest_component(mask))
            np.testing.assert_array_equal(actual['mask'], expected.ravel())
            x0,y0,x1,y1 = _bbox_from_mask(expected)
            self.assertEqual(actual['crop'], dict(x=x0,y=y0,width=x1-x0+1,height=y1-y0+1))

    def test_probability_resize_threshold_and_pad_crop_match_opencv(self):
        rng=np.random.default_rng(108)
        cases=[]
        for mw,mh,w,h in ((9,7,71,91),(71,91,71,91),(13,17,31,41)):
            padded=w+(-w)%4
            values=rng.random((mh,mw), dtype=np.float32)
            cases.append(dict(mw=mw,mh=mh,w=w,h=h,padded=padded,values=values.ravel().tolist()))
        actual=self.node("""
            require('./static/anthropometry-preprocessing.js');
            const fs=require('fs'), cases=JSON.parse(fs.readFileSync(0,'utf8'));
            console.log(JSON.stringify(cases.map(c=>Array.from(AnthropometryPreprocessing.resampleProbability(
              c.values,c.mw,c.mh,c.w,c.h,c.padded)))));
        """,cases)
        for case, result in zip(cases, actual):
            values=np.array(case['values'],np.float32).reshape(case['mh'],case['mw'])
            expected=cv2.resize(values,(case['padded'],case['h']),interpolation=cv2.INTER_LINEAR)[:,:case['w']]>0.5
            np.testing.assert_array_equal(result, expected.ravel())

    def test_browser_refines_crop_and_sends_only_binary_mask_and_points(self):
        result=self.node("""
            const fs=require('fs'),vm=require('vm');
            JSON.parse(fs.readFileSync(0,'utf8'));
            require('./static/anthropometry-preprocessing.js');
            global.window=global;
            const created=[],preview={naturalWidth:71,naturalHeight:91,decode:async()=>{}};
            global.document={getElementById:()=>preview,createElement:()=>{
              const canvas={width:0,height:0};
              const ctx={drawImage:()=>{},createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),
                putImageData:p=>canvas.pixels=p.data};
              canvas.getContext=()=>ctx;
              canvas.toBlob=(callback,type)=>callback({type,pixels:Array.from(canvas.pixels)});
              created.push(canvas);return canvas;
            }};
            let calls=0,closed=0;
            const values=new Float32Array(72*91);
            for(let y=10;y<81;y++)for(let x=20;x<51;x++)values[y*72+x]=1;
            const mask=()=>({width:72,height:91,getAsFloat32Array:()=>values,close:()=>closed++});
            const detector={detect:()=>{
              const crop=calls++%2===1;
              return {landmarks:[Array.from({length:33},()=>({x:crop?0.4:0.1,y:crop?0.6:0.1,visibility:0.9}))],
                segmentationMasks:[mask()]};
            }};
            const html=fs.readFileSync('./templates/index.html','utf8');
            const script=[...html.matchAll(/<script([^>]*)>([\\s\\S]*?)<\\/script>/g)]
              .find(m=>m[2].includes('const MEDIAPIPE_VERSION'))[2];
            vm.runInThisContext(script);
            getLandmarker=async()=>detector;
            prepareAnthropometryInputs({type:'image/jpeg'},{type:'image/jpeg'},()=>{}).then(result=>{
              const landmarks=JSON.parse(result.front.landmarks);
              console.log(JSON.stringify({calls,closed,preprocessing:result.preprocessing,
                cropWidth:created[1].width,cropHeight:created[1].height,
                point:landmarks.left_shoulder,keys:Object.keys(result.front),
                binary:result.front.silhouette.pixels.every(v=>v===0||v===255)}));
            }).catch(e=>{console.error(e);process.exitCode=1;});
        """,{})
        self.assertEqual(result['calls'],4)
        self.assertEqual(result['closed'],4)
        self.assertEqual(result['preprocessing'],'browser-v2')
        self.assertEqual(result['keys'],['silhouette','landmarks'])
        self.assertTrue(result['binary'])
        self.assertGreater(result['cropHeight'],91)
        # The crop is x=16..54, y=0..90; coordinates must map back to 71x91.
        scale=1024/91
        self.assertAlmostEqual(result['point']['x'],(16+0.4*int(39*scale)/scale)/71)
        self.assertAlmostEqual(result['point']['y'],0.6)


if __name__=='__main__':
    unittest.main()
