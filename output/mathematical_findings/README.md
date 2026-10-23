# Short mathematical paper

Upload `mathematical_findings_overleaf.zip` as a new Overleaf project and select `main.tex` as the main document. It is a complete standalone article with an inline bibliography and table. No external figures or bibliography files are required.

Contents:

- `main.tex` / `main.pdf`: mathematical note with full proofs and synthetic examples.
- `claim_review.md`: claim-by-claim scrutiny of the submitted 11-page PDF, including a novelty assessment.
- `perimeter_bounds.py`: reference solver for sharp exact/interval width bounds; stable superellipse helper.
- `verify_math.py` / `verification_results.json`: reproducible synthetic checks and their results.
- `source_manifest.json`: source hashes and build/structural checks.

To run the numerical checks in a Python environment:

```text
python -m pip install numpy scipy clarabel
python verify_math.py
```

The versions used for the recorded results are in `verification_results.json`. Numerical solver outputs are not interval-arithmetic certificates. The paper contains no independently reproduced FAUST, SMPL, BodyM, or human tape experiment. The simple garment-sizing system is excluded.
