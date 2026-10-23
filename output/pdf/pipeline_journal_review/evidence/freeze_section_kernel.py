"""One-time extraction of named historical section functions for verification."""
from pathlib import Path
import ast
import hashlib
import json

here = Path(__file__).resolve().parent
source = Path(r"C:\Users\SUMANTH S SHETTY\Documents\Anthropometry\measurement_extraction.py")
raw = source.read_bytes()
text = raw.decode("utf-8")
tree = ast.parse(text)
names = {"_cross", "graham_scan", "_round_corners", "_adaptive_radius_mm"}
parts = [ast.get_source_segment(text, node) for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(parts) == len(names)
out = here / "historical_section_kernel.py"
out.write_text('"""Frozen named source functions; comments reflect historical implementation, not verified claims."""\nimport numpy as np\nimport cv2\n\n' + "\n\n".join(parts) + "\n", encoding="utf-8")
manifest = {"source_path": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "functions": sorted(names), "extraction": "Python AST named function source segments, unchanged bodies",
            "kernel_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "scope": "2D section morphology/convex hull only; corrected fit supplied separately"}
(here/"section_kernel_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
