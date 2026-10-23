"""Independent reviewer probes; numerical checks, not human validation."""
from pathlib import Path
import importlib.util
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rectifier_conic', root/'evidence'/'corrected_conic.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = []
for ratio in (1.0, 10.0, 100.0):
    for coverage in ('full', 'quarter'):
        angles = np.linspace(0, 2*np.pi if coverage == 'full' else np.pi/2, 100, endpoint=False)
        truth = np.array([160.0, 160.0/ratio])
        rot = np.array([[np.cos(.47), -np.sin(.47)], [np.sin(.47), np.cos(.47)]])
        pts = np.column_stack((truth[0]*np.cos(angles),truth[1]*np.sin(angles))) @ rot.T + [20, -10]
        fit = module.fit_ellipse(pts)
        rows.append({'axis_ratio':ratio, 'coverage':coverage, 'scaled_condition':fit['condition_scaled'], 'max_relative_axis_error':float(np.max(np.abs(fit['axes']/truth-1)))})

angles=np.linspace(0,2*np.pi,50,endpoint=False)
points=np.column_stack((160*np.cos(angles),100*np.sin(angles)))
fit=module.fit_ellipse(points)
covariance_status=module.algebraic_sensitivity(fit)
result={'scope':'Independent exact conic probes of isolated research helper; no human or full pipeline validation.', 'cases':rows, 'exact_fit_sensitivity_status':covariance_status}
(root/'review'/'reviewer_conic_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
