# OneShot-ID Deliverables & Customer Q&A

## 1) Generated deliverables (run: `test0`)

All generated artifacts for this run are stored under:

- `outputs/runs/test0/`

Directory meanings:

- `outputs/runs/test0/candidates/`  
  Stores **all generated person images** (all candidates produced by the generator).

- `outputs/runs/test0/kept/`  
  Stores images that **meet identity-consistency requirements** (accepted outputs).

- `outputs/runs/test0/rejected/`  
  Stores images that **do not meet identity-consistency requirements** (rejected outputs).

- `outputs/runs/test0/faces/`  
  Stores **cropped face images** (largest detected face crops) extracted from generated candidates for review/diagnosis.

- `outputs/runs/test0/reports/`  
  Stores reports that record the **similarity between each generated image and the reference image**, including:
  - `validation_results.csv`
  - `validation_report.md`

## 2) Customer questions: identity consistency & failure modes

### Q: How you ensured identity consistency?

We use InstantID as the main mechanism to preserve identity consistency. The key idea is not to rely on prompts alone, but to inject the reference face as an explicit identity condition into the generation process. That makes the generated person much more stable in terms of facial structure and overall identity.

### Q: Where the system starts to fail?

The system tends to fail when we push the variation too far, especially with strong pose changes such as side profiles, upward or downward angles, exaggerated expressions, or additional occlusions like hands, hats, and masks. In those cases, the visible facial cues become weaker, and the model is more likely to drift away from the original identity.

### Q: What causes identity drift in your setup?

In our setup, the main causes of identity drift are large pose variation and heavy occlusion or expression changes. When the prompt combines things like “side profile + strong expression + occlusion,” the model has to balance identity preservation against the requested variation, and that is where drift usually starts to appear.

### Q: How you would improve identity stability in a production system?

To improve identity stability in a production system, we add a post-generation validation step. We use InsightFace to extract face embeddings from both the reference image and the generated image, and then compute cosine similarity. This gives us a quantitative way to measure identity consistency instead of relying only on visual judgment.

### Q: Why certain outputs should be accepted or rejected?

The accept/reject logic is straightforward: if the similarity score is above the threshold, the output is considered close enough to the reference identity and is saved in the kept folder. If the score is below the threshold, the identity has drifted too far and the image is saved in the rejected folder. This gives the pipeline a clear quality gate and makes the final outputs more reliable and production-friendly.

