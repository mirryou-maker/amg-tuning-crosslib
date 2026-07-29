# Cover letter — ACM Transactions on Mathematical Software

Chun-Yeol You
Department of Physics and Chemistry, DGIST
(Daegu Gyeongbuk Institute of Science and Technology)
Daegu 42988, Republic of Korea
cyyou@dgist.ac.kr

2026-07-29

To the Editor-in-Chief,
ACM Transactions on Mathematical Software

**Re: Submission of the manuscript "Tuning Internal Parameters of Algebraic
Multigrid: A Cross-Library Study of Achievable Speedup and Cheap-Feature
Prediction"**

Dear Editor,

Please consider the enclosed manuscript for publication in ACM Transactions on
Mathematical Software as a research article.

Algebraic multigrid (AMG) is the workhorse preconditioner embedded in every
major sparse-solver framework, yet its performance depends strongly on internal
parameters — coarsening scheme, smoother, and strength-of-connection threshold —
that most users leave at library defaults. Prior algorithm-selection work almost
exclusively chooses *among* solvers or preconditioners, treating an entire AMG
method as a single label. Our manuscript instead opens that label and studies
the interior parameter space empirically. We sweep 88 configurations of AMGCL
over 150 (later 300) SuiteSparse matrices — 13,200 timed solves — and ask two
questions: how much speed is lost by not tuning, and can a good configuration be
predicted from features that are cheap relative to the solve itself.

The main findings are:

- **Tuning matters and requires per-matrix selection.** The oracle speedup over
  the default has median 2.4x (upper quartile 6.4x, maximum 292x); the best
  configuration varies across 14 distinct coarsening/smoother labels, and on 31
  matrices the default fails outright while some configuration succeeds.
- **A predictor works from near-free features, evaluated honestly.** Under a
  strict leave-one-group-out protocol (no sibling matrices leaking across the
  split — a check the literature commonly omits), the predictor picks a
  configuration that solves 83–89% of held-out matrices and captures a median
  43–49% of the achievable speedup. Crucially, near-free structural features
  suffice; spectral features costing 100–1000x more add nothing — a feature-cost
  accounting that prior work seldom performs.
- **A self-critical cross-library validation.** Repeating the study on hypre
  BoomerAMG shows that tunability replicates qualitatively but its magnitude is
  smaller (median 1.6x) and does not transfer between libraries (rank
  correlation 0.24); it is largely governed by default quality — the dramatic
  AMGCL speedups occur precisely where hypre's well-engineered default already
  wins. Intrinsic solvability, by contrast, is library-independent (hard-matrix
  Jaccard 0.76), a split that a predictor-transfer experiment confirms in both
  directions.

We believe the work fits TOMS well on both scope and values. It is an empirical
study of two widely-used pieces of mathematical software (AMGCL and hypre), it
foregrounds honest methodology (feature-cost accounting and group-wise
evaluation), and it is fully reproducible. We release the complete artifact —
the 18,600-solve dataset, the feature extractor, the predictor, and all
figure-generation scripts — publicly at
https://github.com/mirryou-maker/amg-tuning-crosslib . A single command,
`python reproduce.py`, regenerates every figure and headline number from the
released data, and a companion tool, `tools/recommend.py`, applies the predictor
to a user's own matrix. We would welcome consideration under the Replicated
Computational Results process.

We confirm that this manuscript is original, has not been published previously,
and is not under consideration elsewhere. In accordance with the ACM Policy on
Authorship, the manuscript includes a Generative AI disclosure describing the
use of an AI coding assistant (Claude Code, Anthropic) for implementation,
experiment orchestration, and drafting; the author verified all experimental
design, numerical results, and claims against the released code and data, and
takes full responsibility for the content. The work was supported by the
National Research Foundation of Korea (Nos. RS-2025-25463492 and
RS-2026-25472340) and the DGIST R&D Program (26-SR-01).

**Suggested reviewers.** We suggest the following experts in algebraic
multigrid, solver selection, and numerical software (contact details available
on request); we have no conflict of interest with them:

- **Luke N. Olson**, University of Illinois Urbana-Champaign — algebraic
  multigrid and the PyAMG software.
- **Jed Brown**, University of Colorado Boulder — multigrid, PETSc, and
  numerical-software performance.
- **Kanika Sood**, California State University, Fullerton — machine learning
  for iterative-solver selection (the closest prior line of work).
- **Scott P. MacLachlan**, Memorial University of Newfoundland — multigrid
  methods and their analysis.
- **Sivasankaran Rajamanickam**, Sandia National Laboratories — sparse linear
  algebra, performance, and learning-based methods for solvers.

Thank you for considering our submission.

Sincerely,

Chun-Yeol You
Department of Physics and Chemistry, DGIST
cyyou@dgist.ac.kr
