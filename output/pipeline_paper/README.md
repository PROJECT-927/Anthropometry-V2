# Standalone pipeline manuscript

Upload `main.tex` into an empty Overleaf project, or upload the provided ZIP as a project.
Set `main.tex` as the Main document and choose pdfLaTeX. Recompile twice if references
are initially unresolved. This is a complete document, not a section to paste into
another document. It needs no images, .bib file, custom class, or additional .tex files.

The paper contains the implementation pipeline and its mathematical background.
It does not include the separate Fourier-identifiability / multi-view LP paper.
It uses a standard single-column article layout because no target journal is selected.

The pilot tables are inherited observations, not a new end-to-end evaluation.
The paper explicitly identifies code/mathematics discrepancies requiring correction
and re-evaluation. Author details and submission-specific declarations remain to be completed.

The source/model hashes in source_manifest.json identify the paper-adjacent historical
implementation inspected for this manuscript. Application source files were not modified.

Local validation: compiled successfully with Tectonic 0.17.0 (XeTeX), producing 27 pages.
There are no unresolved references, missing characters, or overfull boxes in the final log.
A compiled PDF preview is supplied separately.
