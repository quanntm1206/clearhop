# Loss Refinement Implementation Plan

1. Add failing tests for a locked `refine` stage and paired-control promotion.
2. Add the immutable beta-0.015 config and minimal runner support.
3. Run focused and full tests; run a zero-side-effect dry-run.
4. Screen the refinement arm on seeds 17, 29, and 43.
5. Promote only if the original gates pass; otherwise stop loss refinement.
6. If promoted, run full confirmation and the existing test/export/verify flow.
