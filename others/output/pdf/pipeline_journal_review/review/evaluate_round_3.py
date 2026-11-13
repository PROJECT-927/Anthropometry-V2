"""Record independent final re-review against frozen v2 and unchanged criteria."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parents[1]
review=root/'review'
prior=json.loads((review/'issues_round_2.json').read_text(encoding='utf-8'))
audit=json.loads((review/'build_v2.json').read_text(encoding='utf-8'))
sha=hashlib.sha256((review/'revision_v2.tex').read_bytes()).hexdigest()
assert sha=='136775f3f8c53998e0b9c01d51a9662ece76797b38b638e71c06285f9ba6efac'
assert audit['source_sha256']==sha
issues=prior['issues']
for issue in issues:
    issue['round_3_assessment']=issue['round_2_assessment']
    if issue['id']=='R03':
        issue['round_3_assessment']='The requested 32-case section-morphology/fitted-perimeter experiment has now been performed, with every attempt and four rejected unrounded rectangles preserved. Eight selected cases and source/kernel hashes were independently reproduced. However, the 0.9 opening changes known-ellipse perimeter error from -0.974 to -44.918 mm at 2 mm spacing and from -2.734 to -37.774 mm at 4 mm. The manuscript accurately reports this adverse result and does not retune a default to these cases. R03 remains PARTIALLY ADDRESSED because this exposes an unresolved measurement-prior problem; complete waist/girth/integrated ablations and demonstrated physical utility remain absent.'
    elif issue['id']=='R04':
        issue['round_3_assessment']='The author confirmed that historical reference measurements used a physical tape measure on volunteers, and the frozen manuscript accurately records that fact. This establishes instrument and volunteer participation. It does not establish matched anatomical levels/endpoints, tape tension, operator, posture or repeated readings. The branch/endpoint table and proposed protocol remain useful, but exact matched reference procedure remains unresolved. R04 stays PARTIALLY ADDRESSED.'
    elif issue['id']=='R09':
        issue['round_3_assessment']='The bounded deletion, no-op, off-centre response and identifiability counterexample remain correctly described. The new morphology experiment strengthens the need to separate section rounding from waist correction, rather than supplying evidence that waist deletion is anatomical. R09 remains PARTIALLY ADDRESSED pending justified prior selection and matched waist/morphology evaluation.'
    elif issue['id']=='R12':
        issue['round_3_assessment']='The local computational package now also freezes historical section morphology/Graham functions with source hashes and preserves all 32 section attempts. Historical kernel and section hashes were independently checked. This improves local reproducibility, but an integrated corrected pipeline release with traceable human runs remains unprovided and explicitly unclaimed.'
    elif issue['id']=='R15':
        issue['round_3_assessment']='The new author confirmation of volunteer participation and physical-tape measurement is acknowledged. Volunteer participation alone does not establish documented informed consent, privacy/publication permission or applicable ethics review/exemption. Authorship, affiliation/contact, funding/conflicts and governance facts remain unprovided. Accurate missing-information declarations cannot replace those records. R15 stays OPEN.'
    elif issue['id']=='R21':
        issue['status']='CLOSED'
        issue['round_3_assessment']='Genuine prose consolidation retains implementation settings, equations, adverse findings and declarations. The frozen v2 build is 11 pages including 21 cited references in ascending first-use order, at unchanged 11-point type and 24-mm margins, with zero overfull/underfull, undefined-reference/citation or missing-character warnings. The near-empty twelfth references page is eliminated. Rendering remains the coordinator visual-QA responsibility; the source/build defect and requested pagination change are resolved.'
    else:
        issue['round_3_assessment']+=' This disposition is unchanged after reading the complete v2 manuscript.'
counts={s:sum(i['status']==s for i in issues) for s in ('CLOSED','PARTIALLY ADDRESSED','OPEN')}
assert counts=={'CLOSED':11,'PARTIALLY ADDRESSED':9,'OPEN':2}
result={
 'review_date':'2026-10-21','reviewer':'journal_reviewer (independent agent)',
 'manuscript':'review/revision_v2.tex','source_sha256':sha,
 'verdict':'NOT SATISFIED / REITERATE AFTER SUBSTANTIVE EVIDENCE',
 'submission_readiness':'NOT READY',
 'editorial_and_scoped_numerical_assessment':'Satisfactory within the stated research-reference/synthetic scope; not an overall acceptance PASS.',
 'counts':counts,'new_issues':[],
 'acceptance_contract_unchanged':True,
 'why_no_further_cosmetic_loop':'No identified remaining prose or isolated algebra edit can establish anatomical validity, remove the demonstrated section-prior bias, integrate and evaluate the method, or supply factual author/governance records.',
 'independent_evidence':['reviewer_conic_probe.py','reviewer_conic_probe.json','reviewer_profile_probe.py','reviewer_profile_probe.json','reviewer_section_probe.py','reviewer_section_probe.json'],
 'issues':issues,
}
(review/'issues_round_3.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

header='''# Independent journal re-review — round 3 / final supplied-evidence decision

**Frozen version:** `revision_v2.tex` / `revision_v2.pdf`.  
**Source SHA-256:** `136775f3f8c53998e0b9c01d51a9662ece76797b38b638e71c06285f9ba6efac`.  
**Overall recommendation: NOT SATISFIED / REITERATE AFTER SUBSTANTIVE EVIDENCE.**  
**Submission readiness: NOT READY.**

This decision follows the same acceptance contract as round 1. No new rejection criterion or issue ID is introduced in this round. The complete v2 manuscript, its bibliography, the round-2 response, added section evidence and v2 build audit were read. The manuscript has improved substantially and accurately limits its conclusions to the computations actually tested. An overall PASS would still be unsupported.

The author's late clarification that the historical references used a physical tape measure on volunteers was reviewed before this decision. It is accurately added to the frozen manuscript and documented in `author_clarifications.md`. This establishes the instrument and volunteer participation; it does not establish matched endpoints/repeat procedure, independent recruitment, documented consent or ethics status. R04 and R15 retain their unresolved portions for those reasons.

## What the final rectification achieved

The corrected scaled-SVD conic reference uses the proper completed-square constant and shared recovery for fitting/sensitivity. It passes exact and noisy synthetic checks, including additional independent referee probes. Unavailable sensitivity is represented explicitly. Historical human outcomes are retained in the archive and are no longer used as evidence for the corrected method. Unsupported physical uncertainty components are excluded. Endpoint/source/fallback distinctions and capture/model assumptions are clearer.

The profile experiments now include adverse geometries and resolution dependence. The final requested morphology/fitted-perimeter ablation adds 32 fixed cases with all failures retained. The final source is concise without suppressing implementation or adverse evidence. The coordinator build audit confirms 11 pages including 21 relevant references in ascending first-citation order, unchanged 11-point type/24-mm margins, and zero tracked TeX warnings.

## Independent check of the new evidence

`reviewer_section_probe.py/.json` reproduces eight selected morphology cases and verifies the historical source and frozen section-kernel hashes. Known-ellipse bypass errors are -0.974 mm at 2 mm spacing and -2.734 mm at 4 mm spacing. With opening fraction 0.9, they become -44.918 and -37.774 mm. Four unrounded rectangle attempts fail the five-hull-vertex condition; 0.9-opened rectangles differ from the true rectangular perimeter by -137.093/-131.257 mm. These comparisons combine grid/morphology/hull/conic/perimeter effects, as the paper correctly states. They neither identify a universal replacement parameter nor explain the old human errors causally.

The adverse finding is scientifically material. It means that correcting conic algebra alone does not make the complete measurement method trustworthy: a strong section prior can substantially alter girth even on a known ellipse. This remains within R03/R09, which already required separating morphology from claimed adaptation benefit. It is not a newly invented standard.

Earlier independent probes also reproduced selected profile improvements and failures, verified frozen source hashes and boundary-tolerance tests, and checked exact conic cases. All probes have their code and outputs in the review folder. No referee probe is a full perception-to-measurement run or a human agreement study. The coordinator performs rendered-page visual inspection; this report does not claim independent inspection of every PDF page.

## Final issue-by-issue disposition

'''
body='\n'.join(f"### {i['id']} — {i['title']}\n\n**Status: {i['status']}.** {i['round_3_assessment']}\n" for i in issues)
footer='''
## What prevents acceptance

**Final count: 11 CLOSED, 9 PARTIALLY ADDRESSED, 2 OPEN.** Closed issues include truthful removal of claims that could not be validated; this is not equivalent to performing the excluded experiments. The two open items are R02 independent corrected human validation and R15 factual authorship/governance/declaration information. The partially addressed items contain substantive method and reproducibility requirements, not merely paperwork.

The strongest remaining scientific obstacle is the unsupported utility of the complete geometric method. The 0.9 rounding prior needs reassessment/justification, profile filtering can suppress genuine depth variation, and waist deletion has no demonstrated anatomical identification. The paper cannot currently establish that these choices improve anthropometric measurement or constitute an independently useful advance over established approaches. Its careful wording makes the report honest but does not supply that result.

Before another scientific acceptance review, the authors should provide:

1. A justified and tested section-measurement/prior strategy addressing the observed rounding bias, with complete matched waist/morphology/girth and profile ablations. Do not select a new default solely to fit these synthetic truths.
2. A frozen integrated corrected pipeline, reproducible environment/model/configuration manifests, endpoint/method/source records, and all attempted runs and rejection reasons.
3. Appropriate independent observations with matched tape definitions, capture/repeat records and participant-level analysis, separated from development/tuning cases; meaningful comparisons to relevant methods are needed for an advantage claim.
4. Actual authorship/affiliation/contact and applicable consent/privacy/ethics, funding/conflict and data-access declarations. None may be inferred or invented.

Further cosmetic revisions would not change the overall decision. An honest stopping point is to deliver this improved internal manuscript, computational evidence and complete unresolved-evidence list, then re-review when substantive evidence is available. This report is an independent simulated journal review, not a decision by an external journal editor.
'''
(review/'round_3_review.md').write_text(header+body+footer,encoding='utf-8')

decision='''# Final independent reviewer decision

**NOT SATISFIED — not ready for journal submission.**

Reviewed frozen `revision_v2.tex/.pdf`, source SHA-256 `136775f3f8c53998e0b9c01d51a9662ece76797b38b638e71c06285f9ba6efac`.

There were 22 tracked issues: **11 closed, 9 partially addressed, 2 open** after the reviewer–rectifier–reviewer cycle and one targeted additional revision. The 11-page manuscript meets the requested <=12-page limit and has 21 references in ascending order of first citation. The numerical/editorial improvements are satisfactory within the explicit synthetic research-reference scope; no overall PASS is awarded.

The conic sign error was corrected and tested in an isolated reference. Historical human estimates were archived and excluded from corrected-method evidence. This resolves misleading presentation; it does not create corrected human validation.

The final experiment found a separate material problem: the 0.9 section-opening prior underestimates the perimeter of a known ellipse by **44.92 mm** at 2 mm spacing, compared with **0.97 mm** when opening is bypassed. At 4 mm the corresponding errors are 37.77 versus 2.73 mm. The paper now reports this. A replacement default cannot be justified from those test cases alone; the section prior and complete measurement strategy require reassessment and evaluation.

Acceptance still requires a justified integrated method, independent matched anthropometric validation, meaningful component/prior-method comparisons, a reproducible corrected pipeline and factual author/governance documentation. The profile and waist priors have unvalidated physical benefit. Missing consent/ethics or study facts have not been fabricated.

The author's confirmation of **physical tape measurements on volunteers** was incorporated before this decision. It confirms instrument and volunteer participation, but leaves the exact tape protocol, matched endpoints, independence and documented consent/ethics status unresolved.

The complete reasons and exact resolution criteria are in `round_1_review.md`, `round_1_addendum_R22.md` and `issues_round_1.json`/`issue_R22.json`; every final status and reason is in `round_3_review.md` and `issues_round_3.json`. Independent reproducibility probes are preserved beside them.

No remaining identified wording change can honestly convert this recommendation to PASS. Re-review should follow the substantive evidence listed in the final report. This is a simulated independent review, not an external journal acceptance decision.
'''
(review/'final_decision.md').write_text(decision,encoding='utf-8')
print(json.dumps({'verdict':result['verdict'],'counts':counts,'source_sha256':sha},indent=2))
