# Condensed pipeline manuscript

The compiled manuscript is **9 A4 pages, including the references**, within the requested 12-page maximum. Body text is 11 pt with 24 mm margins. This is a rewritten version of the preceding 23-page pipeline paper.

## Files

- `main.tex`: complete standalone article with an embedded bibliography. It begins with `\documentclass` and includes `\begin{document}` and `\end{document}`.
- `main.pdf`: locally compiled and visually checked manuscript.
- `pipeline_paper_overleaf.zip`: upload-ready archive containing `main.tex` and this README.
- `editorial_notes.md`: scope of the condensation and scientific issues retained.
- `validation.json`: page-count, compilation, and citation-order checks.
- `source_manifest.json`: source and output hashes, with the preceding manuscript's implementation provenance.

## Overleaf

Create a new project using **Upload Project** and select `pipeline_paper_overleaf.zip`. Alternatively, put the entire `main.tex` file in an empty project and set it as the main document. Select pdfLaTeX. There are no external images, bibliography databases, custom classes, or included source files. The bibliography needs no BibTeX step.

Compilation was checked locally with Tectonic/XeTeX. A live Overleaf compile was not performed. Author details remain a placeholder.

## Scope

The implementation and mathematical formulation are retained in the main text. The module inventory, command-line appendix, duplicate descriptions, and secondary pilot tables were removed. All 20 bibliography entries are cited, with first appearances numbered consecutively from 1 to 20.

The historical conic-recovery discrepancy and sensitivity limitations remain disclosed. The pilot observations were not recomputed. This manuscript describes the historical implementation identified in the source manifest; it does not assert that later application revisions reproduce those records.
