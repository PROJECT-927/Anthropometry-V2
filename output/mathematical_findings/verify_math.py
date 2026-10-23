"""Reproduce synthetic numerical checks; no human dataset is used."""
from pathlib import Path
import sys, json, time
import numpy as np
from scipy.special import ellipe
from scipy.spatial import ConvexHull
import clarabel, scipy
from perimeter_bounds import (perimeter_bounds, closed_upper,
                              stable_superellipse_perimeter, polygon_perimeter)

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261018)
start = time.perf_counter()
two_errors, gaps, statuses = [], [], []
for _ in range(40):
    w, d = rng.uniform(40, 500, 2)
    result = perimeter_bounds([0, np.pi/2], [w, d])
    two_errors.append(abs(result['lower']-2*np.hypot(w, d)))
    assert abs(result['upper']-2*(w+d)) < 1e-7
    gaps.append(result['duality_gap']); statuses.append(result['status'])
assert max(two_errors) < 1e-4
containment_violations, upper_discrepancies = [], []
for i in range(40):
    pts = rng.normal(size=(20, 2)) @ np.diag(rng.uniform(30, 140, 2))
    pts = pts[ConvexHull(pts).vertices]
    p = polygon_perimeter(pts)
    k = [2, 3, 4, 6, 8][i%5]
    angles = np.sort(rng.uniform(0, np.pi, k))
    while np.min(np.diff(np.r_[angles, angles[0]+np.pi])) < 0.08:
        angles = np.sort(rng.uniform(0, np.pi, k))
    u = np.column_stack((np.cos(angles), np.sin(angles)))
    widths = np.ptp(pts @ u.T, axis=0)
    result = perimeter_bounds(angles, widths)
    violation = max(result['lower']-p, p-result['upper'], 0)
    containment_violations.append(violation)
    upper_discrepancies.append(abs(result['upper']-closed_upper(angles, widths)))
    gaps.append(result['duality_gap']); statuses.append(result['status'])
    full = np.vstack((result['contacts'], -result['contacts']))
    assert np.max(abs(np.ptp(full @ u.T, axis=0)-widths)) < 1e-4
    maxgap = np.max(np.diff(np.r_[angles, angles[0]+np.pi]))
    assert result['lower'] >= np.cos(maxgap/2)*result['upper']-1e-4
assert max(containment_violations) < 1e-4
assert max(upper_discrepancies) < 1e-6
table = []
for k in [2, 3, 4, 6, 8]:
    angles = np.arange(k)*np.pi/k
    widths = 2*np.sqrt(165**2*np.cos(angles)**2+112**2*np.sin(angles)**2)
    result = perimeter_bounds(angles, widths)
    table.append(dict(k=k, lower_mm=result['lower'], upper_mm=result['upper'],
                      true_mm=4*165*ellipe(1-(112/165)**2),
                      optimal_uniform_relative_bound=float(np.tan(np.pi/(4*k))**2)))
    gaps.append(result['duality_gap']); statuses.append(result['status'])
noisy = perimeter_bounds([0, np.pi/2], [328, 222], [332, 226])
assert abs(noisy['lower']-2*np.hypot(328, 222)) < 1e-4
assert abs(noisy['upper']-2*(332+226)) < 1e-6
try:
    perimeter_bounds([0, np.pi/4, np.pi/2], [100, 200, 100])
except ValueError:
    inconsistent_rejected = True
else:
    raise AssertionError('Inconsistent widths were accepted.')
family = {str(n):stable_superellipse_perimeter(330,224,n) for n in [1,2,10,200,float('inf')]}
assert abs(family['2']-4*165*ellipe(1-(112/165)**2)) < 1e-7
assert all(x<y for x,y in zip(family.values(),list(family.values())[1:]))
hard_cases = []
for k in [2,3,4,6,8]:
    angles = np.arange(k)*np.pi/k
    breaks = np.sort(np.mod(angles+np.pi/2,np.pi))
    breaks = np.r_[breaks, breaks[0]+np.pi]
    j = np.argmax(np.diff(breaks))
    phi = (breaks[j]+breaks[j+1])/2
    widths = 100*np.abs(np.cos(angles-phi))
    result = perimeter_bounds(angles,widths)
    expected_upper = 200/np.cos(np.pi/(2*k))
    assert abs(result['lower']-200) < 1e-4
    assert abs(result['upper']-expected_upper) < 1e-6
    hard_cases.append(dict(k=k,lower_mm=result['lower'],upper_mm=result['upper'],
                           expected_relative_risk=float(np.tan(np.pi/(4*k))**2)))
report = {
    'seed':20261018,'human_datasets_used':False,
    'two_view_closed_form_cases':40,'max_two_view_lower_error_mm':max(two_errors),
    'random_polygon_cases':40,'max_containment_violation_mm':max(containment_violations),
    'max_upper_polygon_formula_difference_mm':max(upper_discrepancies),
    'max_primal_dual_gap_mm':max(gaps),'solver_status_counts':{s:statuses.count(s) for s in set(statuses)},
    'ellipse_example':table,'superellipse_perimeters_mm':family,
    'sharp_view_design_witnesses':hard_cases,
    'interval_two_view':{k:noisy[k] for k in ['lower','upper','duality_gap']},
    'inconsistent_widths_rejected':inconsistent_rejected,
    'zero_embedded_difference_counterexample':{'R_mm':100,'a_mm':1,'width_samples_mm':[100]*4,
        'Q2_mm':float(np.pi*100),'Q4_mm':float(np.pi*100),'P_mm':float(np.pi*99),
        'embedded_difference_mm':0,'actual_error_mm':float(np.pi)},
    'elapsed_seconds':time.perf_counter()-start,
    'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'clarabel':clarabel.__version__},
    'interpretation':'Floating-point checks support implementation consistency; proofs establish the mathematical statements.'}
(OUT/'verification_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
