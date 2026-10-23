"""Isolated research pipeline: explicit candidate outputs, no truth-based selection.

This driver does not modify the application and does not claim that development
captures validate physical accuracy. Silhouette mode bypasses perception and
uses declared stature-proportion level proxies, recorded in every result.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE/'source'))
sys.path.insert(0,str(ROOT/'evidence'))
import cv2
import numpy as np
from scipy import ndimage
from visual_hull import visual_hull,compute_scale_mm_per_px,body_center_x
from profile_carve import carve_from_front_profile
from waist_correction import find_waist
from waist_policy_v3 import propose_displacement,evaluate_hypothesis
from candidate_diagnostics_v3 import augment_candidate_diagnostics
from measurement_extraction import _torso_bounds_at,measure_shoulder_arc
from section_candidates_v3 import measure_section,compare_waist_hypothesis
from profile_ablation import analyze_levels

VERSION='integrated-candidates-v3.0'
EXPECTED_MODEL_SHA='64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b'
POLICY={
 'version':VERSION,'candidate_selection':'report_all_named_candidates_no_oracle_selection',
 'base_waist_deletion':False,'base_section_opening':False,'base_section_closing':False,
 'whole_volume_component_deletion':False,'whole_volume_extent_smoothing':False,
 'conditional_depth_cap':False,'front_torso_clip_multiplier':1.15,
 'fragment_rejection_fraction':0.01,'mask_threshold':0.5,'minimum_frame_foreground_fraction':0.02,
 'silhouette_level_fractions_from_crown':{'shoulder':0.182,'hip':0.470,'ankle':0.961},
 'waist_shoulder_hip_fraction':0.688,'chest_shoulder_waist_fraction':0.35,
 'profile_membership_epsilon_over_h':1e-8,
 'sensitivity':'unavailable_not_calibrated; no legacy RSS',
 'supplied_mask_preprocessing':'grayscale>127; nonempty/border guard only; no component removal or hole filling',
 'integrated_outputs':'chest, waist proxy, hip girth candidates only; shoulder specified separately not run',
 'display_export':'not executed; measurements do not depend on a mesh',
}


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_peak():
    if sys.platform!='win32':return {'available':False,'reason':'Windows peak-memory probe only'}
    class MemoryCounters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
          ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
           'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
    info=MemoryCounters();info.cb=ctypes.sizeof(info)
    ok=psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb)
    return {'available':bool(ok),'peak_working_set_bytes':int(info.PeakWorkingSetSize),
            'peak_pagefile_bytes':int(info.PeakPagefileUsage),'method':'Windows GetProcessMemoryInfo; process lifetime'}


def border(mask):
    return bool(mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any())


def dominant_mask(binary):
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
    if n<=1:raise ValueError('empty_foreground')
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    kept=labels==largest;removed=int(binary.sum()-kept.sum())
    fraction=removed/max(int(binary.sum()),1)
    if fraction>POLICY['fragment_rejection_fraction']:
        raise ValueError('ambiguous_foreground_components_material_fraction')
    return kept.astype(np.uint8),{'components':int(n-1),'removed_pixels':removed,'removed_fraction':fraction}


def raw_view(path,model):
    import mediapipe as mp
    import mediapipe_silhouette as segmentation
    import pose_normalization as pose
    pose._POSE_MODEL_PATH=str(Path(model).resolve())
    image=cv2.imread(str(path))
    if image is None:raise ValueError('image_unreadable')
    height,width=image.shape[:2];pad=(-width)%4
    padded=cv2.copyMakeBorder(image,0,0,0,pad,cv2.BORDER_REPLICATE) if pad else image
    rgb=np.ascontiguousarray(cv2.cvtColor(padded,cv2.COLOR_BGR2RGB))
    detection=segmentation._detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    if not detection.pose_landmarks:raise ValueError('missing_pose')
    if not detection.segmentation_masks:raise ValueError('missing_segmentation')
    probability=np.squeeze(detection.segmentation_masks[0].numpy_view())
    if probability.shape[:2]!=padded.shape[:2]:
        probability=cv2.resize(probability,(padded.shape[1],padded.shape[0]),interpolation=cv2.INTER_LINEAR)
    foreground=(probability[:height,:width]>.5).astype(np.uint8)
    mask,components=dominant_mask(foreground)
    mask=segmentation.fill_small_holes(mask)
    if float(mask.mean())<.02:raise ValueError('foreground_below_minimum_fraction')
    if border(mask):raise ValueError('input_mask_border_contact')
    normalized,info=pose.normalize_pose(image,mask)
    # Test the transformed original support, before output-canvas clipping.
    rows,cols=np.where(mask)
    points=np.column_stack((cols,rows))
    transformed=pose.apply_similarity_to_points(points,info['R'],info['s'],info['t'])
    lost=((transformed[:,0]<-.5)|(transformed[:,0]>width-.5)|
          (transformed[:,1]<-.5)|(transformed[:,1]>height-.5))
    if lost.any():raise ValueError('foreground_lost_by_alignment_canvas')
    if border(normalized):raise ValueError('normalized_mask_border_contact')
    record={'image_shape':list(image.shape),'padding_columns':pad,'components':components,
            'rotation_deg':info['rotation_deg'],'scale_transform':info['s'],
            'alignment_out_of_canvas_source_pixels':int(lost.sum()),
            'landmarks':info['landmarks'],'landmarks_normalized':info['landmarks_normalized'],
            'foreground_pixels':int(mask.sum()),'normalized_pixels':int(normalized.sum())}
    return normalized,info['landmarks_normalized'],record


def silhouette_view(path):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError('mask_unreadable')
    mask=(image>127).astype(np.uint8)
    if not mask.any():raise ValueError('empty_foreground')
    if border(mask):raise ValueError('input_mask_border_contact')
    return mask,{'image_shape':list(image.shape),'provided_silhouette':True,'foreground_pixels':int(mask.sum())}


def clipped_section(hull,y,front_mask,scale,top,centre):
    index=int(np.argmin(abs(hull['y_mm']-y)))
    lo,hi=_torso_bounds_at(front_mask,float(hull['y_mm'][index]),top,scale,centre)
    lo*=1.15;hi*=1.15
    mask=hull['occupied'][:,index,:].copy()
    mask&=(hull['x_mm'][:,None]>=lo)&(hull['x_mm'][:,None]<=hi)
    return mask,index,{'requested_level_mm':float(y),'sampled_level_mm':float(hull['y_mm'][index]),
                       'front_torso_bounds_mm':[float(lo),float(hi)]}


def execute(args):
    started=time.perf_counter();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    result={'version':VERSION,'attempt_id':args.attempt_id,'evidence_class':args.evidence_class,
            'input_mode':args.mode,'height_cm':args.height_cm,'status':'rejected','reason':None,
            'policy':POLICY,'stages':[],'inputs':{},
            'runtime_environment':{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor()},
            'sections':{},'physical_uncertainty':{'available':False,'value_mm':None,'reason':'not_calibrated'}}
    def stage(name,fn):
        start=time.perf_counter()
        try:return fn()
        finally:result['stages'].append({'stage':name,'elapsed_seconds':time.perf_counter()-start})
    try:
        if not np.isfinite(args.height_cm) or args.height_cm<=0:raise ValueError('invalid_stature')
        result['inputs']={'front_sha256':digest(args.front),'side_sha256':digest(args.side)}
        if args.mode=='rgb':
            if not args.model or digest(args.model)!=EXPECTED_MODEL_SHA:raise ValueError('missing_or_unpinned_pose_model')
            result['model_sha256']=EXPECTED_MODEL_SHA
            front,lm,front_info=stage('front_perception',lambda:raw_view(args.front,args.model))
            side,side_lm,side_info=stage('side_perception',lambda:raw_view(args.side,args.model))
            result['landmark_source']='pretrained_pose_on_RGB'
        else:
            front,front_info=stage('front_supplied_mask',lambda:silhouette_view(args.front))
            side,side_info=stage('side_supplied_mask',lambda:silhouette_view(args.side))
            lm=side_lm=None
            result['landmark_source']='stature_proportion_level_proxies; perception_and_pose_bypassed'
        result['views']={'front':front_info,'side':side_info}
        np.savez_compressed(output/'normalized_masks.npz',front=front,side=side)
        sf,tf,bf=compute_scale_mm_per_px(front,args.height_cm);cx=body_center_x(front,tf,bf)
        ss,ts,bs=compute_scale_mm_per_px(side,args.height_cm);cs=body_center_x(side,ts,bs)
        if lm is not None:
            def level(pair):return float((sum(lm[k][1] for k in pair)/2-tf)*sf)
            shoulder=level(('left_shoulder','right_shoulder'));hip=level(('left_hip','right_hip'))
        else:
            shoulder=.182*args.height_cm*10;hip=.470*args.height_cm*10
        waist=shoulder+.688*(hip-shoulder);chest=shoulder+.35*(waist-shoulder)
        result['levels_mm']={'shoulder':shoulder,'chest':chest,'waist_proxy':waist,'hip_landmark_or_proxy':hip}
        conventional=stage('conventional_reconstruction',lambda:visual_hull(front,side,args.height_cm,args.voxel_mm,args.coarse_mm))
        branches={'conventional':conventional}
        try:branches['profile']=stage('profile_reconstruction',lambda:carve_from_front_profile(front,side,args.height_cm,args.voxel_mm))
        except Exception as exc:result['profile_failure']={'type':type(exc).__name__,'reason':str(exc)}
        result['reconstruction']={name:{'grid_shape':list(h['occupied'].shape),'voxel_mm':h['voxel_mm'],
            'occupied_voxels':int(h['occupied'].sum()),'component_deletion':False,
            'profile_rejected_rows':h.get('rejected_rows'),'profile_rows':h.get('profile_rows_total')} for name,h in branches.items()}
        def sections():
            for site,y in [('chest_level',chest),('waist_proxy',waist),('hip_level',hip)]:
                result['sections'][site]={}
                for branch,hull in branches.items():
                    mask,j,metadata=clipped_section(hull,y,front,sf,tf,cx)
                    metadata.update({'site':site,'source_branch':branch,'waist_deletion':False,
                      'profile_interpolated':bool(hull['depth_interpolated'][j]) if branch=='profile' else None})
                    record=measure_section(mask,hull['x_mm'],hull['z_mm'],metadata)
                    result['sections'][site][branch]=augment_candidate_diagnostics(record,mask,hull['x_mm'],hull['z_mm'])
            return None
        stage('section_candidates',sections)
        result['level_ablations']=stage('level_ablations',lambda:analyze_levels(front,side,args.height_cm,{'chest_level':chest,'waist_proxy':waist,'hip_level':hip},args.voxel_mm))
        def waist_hypothesis():
            trace=find_waist(front,shoulder,hip,tf,sf,cx)
            wi=int(np.argmin(abs(trace['y_mm']-waist)))
            left=propose_displacement(trace['y_mm'],trace['half_width_left_mm'],wi)
            right=propose_displacement(trace['y_mm'],trace['half_width_right_mm'],wi)
            out={'left_proposal':left,'right_proposal':right,'default_applied':False}
            if left['status']!='proposed' or right['status']!='proposed':
                out['status']='not_admitted';return out
            out['evaluation']=evaluate_hypothesis(conventional,float(trace['y_mm'][wi]),
                left['displacement_mm'],right['displacement_mm'],
                float(trace['half_width_left_mm'][wi]),float(trace['half_width_right_mm'][wi]),
                metadata={'level_source':'fixed_fraction_trace_nearest_row','waist_deletion':True})
            return out
        try:result['optional_waist_hypothesis']=stage('optional_waist_hypothesis',waist_hypothesis)
        except Exception as exc:result['optional_waist_hypothesis']={'status':'unavailable','reason':str(exc)}
        result['status']='computed_candidate_set'
    except Exception as exc:
        result['reason']=str(exc);result['exception_type']=type(exc).__name__
        result['traceback']=traceback.format_exc()
    result['elapsed_seconds']=time.perf_counter()-started
    result['process_memory']=process_peak()
    result['source_hashes']={str(p.relative_to(HERE)):digest(p) for p in sorted((HERE/'source').glob('*.py'))}
    result['candidate_module_sha256']=digest(ROOT/'evidence/section_candidates_v3.py')
    result['driver_sha256']=digest(__file__)
    (output/'result.json').write_text(json.dumps(result,indent=2,default=serial)+'\n',encoding='utf-8')
    print(json.dumps({'attempt_id':args.attempt_id,'status':result['status'],'reason':result['reason'],
                      'elapsed_seconds':result['elapsed_seconds'],'peak_memory':result['process_memory']},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--front',required=True);parser.add_argument('--side',required=True)
    parser.add_argument('--height-cm',required=True,type=float);parser.add_argument('--attempt-id',required=True)
    parser.add_argument('--mode',choices=['rgb','silhouettes'],default='rgb')
    parser.add_argument('--evidence-class',required=True);parser.add_argument('--model')
    parser.add_argument('--output',required=True);parser.add_argument('--voxel-mm',type=float,default=2.)
    parser.add_argument('--coarse-mm',type=float,default=10.)
    execute(parser.parse_args())
