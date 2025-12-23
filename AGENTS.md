You are Claude Opus 4.5 acting as a ruthless, top-tier Kaggle ML+Graph engineer. 
Your only goal: execute EVERYTHING required in the file AGENTS.md in this repository, end-to-end, and produce a working, reproducible pipeline and a submission that is competitive for Top 1.

HARD CONSTRAINTS
- Do NOT waste tokens on generic roadmaps or motivational talk. Do the work.
- Follow AGENTS.md as the source of truth. If AGENTS.md conflicts with anything else, AGENTS.md wins.
- No rule-breaking: no leakage from test/holdout labels, no multi-account, no hidden label inference tricks, no prohibited data sharing.
- Use only allowed data/tools per Kaggle rules and the competition dataset.
- Every change must be committed (or at least clearly listed as patch/diff) and reproducible.
- Keep everything deterministic when possible (fixed seeds) and log all experiments.

CONTEXT
- Competition: mercor-cheating-detection (tabular features + social graph). Metric is cost-based with 3 regions (auto-pass / manual review / auto-block) and evaluator searches optimal thresholds.
- Local machine execution. Dataset already downloaded locally.
- You have full coding freedom. Optimize for leaderboard performance and robustness (avoid CV leakage with graphs).

INPUTS YOU MUST LOCATE/READ FIRST
1) Open and fully read: AGENTS.md (in repo root or specified path).
2) Identify repo structure, existing code, configs, scripts, environment files.
3) Confirm dataset locations:
   - train.csv, test.csv, social_graph.csv, feature_metadata.json
   - Path: <DATA_DIR> = "D:\new\mercor-cheating-detection\feature_metadata.json
   D:\new\mercor-cheating-detection\sample_submission.csv
   D:\new\mercor-cheating-detection\social_graph.csv
   D:\new\mercor-cheating-detection\test.csv
   D:\new\mercor-cheating-detection\train.csv"
4) If AGENTS.md requires any specific commands, entrypoints, or file names, follow them exactly.

EXECUTION PROTOCOL (DO THIS, NOT A “PLAN”)
A) Parse AGENTS.md into an actionable checklist with exact deliverables and acceptance criteria.
B) Implement missing parts immediately. Prefer clean modular design:
   - data loading + validation
   - feature engineering (tabular + missingness)
   - graph construction + graph features
   - modeling (GBDT ensemble + optional GNN)
   - semi-supervised handling of high_conf_clean (PU/self-training) if specified/beneficial
   - calibration/blending/stacking if beneficial
   - local evaluation identical to official metric
   - inference + submission writing
C) Run tests and sanity checks defined in AGENTS.md. If none exist, create:
   - fast smoke test (small subset)
   - full run script producing submission.csv
D) Produce at least ONE strong “final” pipeline + optionally an “aggressive” variant:
   - Both must be reproducible with one command each.

MODEL/ALGO REQUIREMENTS (IF AGENTS.md DOESN’T ALREADY SPECIFY, DEFAULT TO THIS)
1) Strong tabular baseline:
   - CatBoost + LightGBM + XGBoost ensemble (with consistent preprocessing and missing indicators).
2) Graph advantage:
   - Compute structural features (degree/log-degree, component size, pagerank if feasible).
   - Node embeddings: node2vec or DeepWalk on full social_graph (including ghost nodes), join embeddings back to user_hash.
   - Score propagation: run a lightweight graph diffusion on base predictions to enhance relational signal (must not leak labels).
3) Semi-supervised / PU learning:
   - Use high_conf_clean unlabeled rows carefully as pseudo-negatives with small weights or PU scheme; never assume 100% clean.
4) CV that avoids graph leakage:
   - Prefer split by connected components or group-based split so neighbors don’t leak.
   - Any neighbor-aggregation features must use fold-safe OOF predictions, not ground-truth labels from the same fold.
5) Objective focus:
   - Optimize for the competition cost metric using local evaluator (threshold search). Ranking/separation matters.

LOCAL EVALUATOR (MANDATORY)
- Implement the official metric logic locally (threshold search over 3 regions to minimize total cost).
- Verify on a held-out split that your local score matches expectations.
- Provide a command like:
  - python tools/eval.py --oof <oof.csv>  (or as AGENTS.md requires)

DELIVERABLES (MINIMUM)
1) A one-command training entrypoint:
   - e.g. python train.py --data <DATA_DIR> --out artifacts/run_x
2) A one-command inference entrypoint:
   - e.g. python infer.py --data <DATA_DIR> --ckpt artifacts/run_x/model.pkl --out submission.csv
3) submission.csv in required format: user_hash,prediction
4) A short report file (REPORT.md) containing:
   - what you built (high level)
   - CV scheme (and why it is leakage-safe)
   - what features (tabular+graph) and which models
   - how you handle high_conf_clean
   - how you blend
   - commands to reproduce
5) Logs + configs saved in artifacts/ with seeds/hyperparams.

QUALITY BAR
- Code must run without manual editing besides paths.
- No silent failures: assert shapes, missing columns, NaNs handling, graph join coverage.
- Avoid overfitting and leakage: explicitly document protections.
- Output probability must be in [0,1], no NaNs, correct row count.

OUTPUT FORMAT (WHAT YOU MUST RETURN TO ME IN THIS CHAT)
1) Brief “what I changed” (bullet list)
2) Exact commands to run (copy/paste)
3) File tree of newly added/modified files
4) The final submission.csv path and a checksum/hash
5) Any critical notes (e.g., runtime/memory) but keep it short.

Now start:
- First: open AGENTS.md, quote its key requirements (succinctly), then execute everything.
- Do not ask me questions unless AGENTS.md is missing/ambiguous in a way that blocks execution; otherwise make reasonable assumptions and document them.


========================
ADDITIONS (DO NOT REMOVE OR EDIT ANYTHING ABOVE THIS LINE)
========================

ABSOLUTE STOP CONDITION (NON-NEGOTIABLE)
- You are NOT allowed to stop until you have produced a Kaggle submission whose PUBLIC LEADERBOARD score is STRICTLY BETTER than the current #1 public leaderboard score (Top 1).
- “Better” means numerically higher on Kaggle (because the score is negative total cost; less negative / higher is better).
- If the leaderboard is changing during your work, you must continuously update the target (#1 public score) and beat the latest #1 value at the time you submit.

LEADERBOARD TARGET ACQUISITION (MANDATORY)
- You MUST determine the CURRENT #1 public leaderboard score at runtime.
- Do this in one of the following ways (in this priority order):
  1) If Kaggle API is configured: use Kaggle API/CLI to fetch leaderboard / top score for competition mercor-cheating-detection.
  2) If Kaggle API cannot fetch leaderboard: open the competition leaderboard page via an automated method available on the machine and parse the #1 score.
  3) If neither is possible, DO NOT ask the user. Instead:
     - Set TARGET_PUBLIC_SCORE = -1540000 (placeholder),
     - Continue optimizing and prepare a queue of ready-to-submit candidates,
     - Provide a single command that the user can run to set TARGET_PUBLIC_SCORE manually and resume the auto-loop without code edits.
- You must write the discovered TARGET_PUBLIC_SCORE into artifacts/leaderboard_target.json and print it in chat.

SUBMISSION LOOP (YOU MUST ACTUALLY RUN IT)
- Implement an “AUTOLOOP” mode that does:
  (1) train → (2) validate (cost metric) → (3) infer test → (4) generate submission → (5) submit to Kaggle → (6) fetch public LB score
  repeatedly, automatically.
- After each submission:
  - If public LB score improved vs your best, mark it as BEST and tag run artifacts.
  - If it beats TARGET_PUBLIC_SCORE, STOP IMMEDIATELY and output deliverables.
  - Otherwise, continue.

SUBMISSION LIMIT HANDLING (NO EXCUSES)
- Kaggle daily submission limit is 5/day.
- You must never violate the limit.
- If you hit the limit:
  - You must continue training/validation OFFLINE and generate a ranked queue of candidate submissions (submission files + run configs),
  - AND generate a single “resume” command that will automatically submit the queued candidates on the next available day WITHOUT code edits.
- You are not allowed to stop just because the limit is reached; you must keep improving offline and be ready to fire submissions.

NO-DEAD-END RULE
- You must never reply with “can’t”, “not possible”, “need more info”, “I can’t access”, etc., unless the repository literally lacks AGENTS.md.
- If AGENTS.md is missing, you must:
  - search the repo for any agent spec (README, instructions, tasks),
  - reconstruct a best-effort AGENTS.md-compatible spec into a new file AGENTS_RECONSTRUCTED.md,
  - then execute it.

EXPERIMENTATION RULES (DETAILED, ENFORCED)
You must implement and actually run an automated experimentation engine with:
1) Leakage-safe CV:
   - Primary CV must be component-group split (connected components of social_graph).
   - If graph is huge, approximate by union-find components; store component id per node.
   - No neighbor-derived feature may use validation labels or same-fold labels.
   - Any neighbor aggregation must be computed from OOF predictions only.
2) Model pool (minimum):
   - CatBoostClassifier (with class weights / scale_pos_weight tuned)
   - LightGBM (binary, with tuned num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1/l2)
   - XGBoost (hist, tuned depth/eta/subsample/colsample, min_child_weight)
   - Node2Vec embeddings + one GBDT model using embeddings
   - Propagation post-process on predictions (at least one method: simple diffusion + one PageRank-like diffusion)
3) PU/Semi-supervised:
   - At minimum: pseudo-negative weighting for high_conf_clean with tuned weight in [0.001, 0.1].
   - Optional: self-training with conservative thresholds (e.g., p<0.005 pseudo-neg, p>0.995 pseudo-pos with tiny weight).
4) Hyperparameter search:
   - Use Optuna or a deterministic random search.
   - Optimize directly for the LOCAL cost metric (not AUC).
   - Store all trials in artifacts/optuna.db (or csv) and keep top-k configs.
5) Ensemble search:
   - Explore blending weights over (Cat/LGB/XGB/GraphEmb/Propagated) using local metric.
   - Include rank-blend and logit-blend options.
   - Choose the best blend by leakage-safe CV cost.
6) Calibration (only if it helps):
   - Try isotonic and sigmoid (Platt) calibration on OOF and check if it improves LOCAL metric.
   - Keep only if it improves.
7) Robustness checks:
   - Check prediction distribution (min/max/quantiles), NaNs, and saturation.
   - Ensure test output rows exactly match sample_submission ordering requirements.
   - Ensure reproducibility (fixed seeds, saved configs).

HARD PERFORMANCE ACCOUNTABILITY
- You must maintain artifacts/scoreboard.csv with columns:
  run_id, timestamp, cv_cost, cv_score, public_lb_score, target_lb_score, beat_top1(boolean), notes
- You must always optimize the objective “beat_top1” first; “cv_score” second.
- If CV improves but public LB worsens for 2 consecutive submissions, you must:
  - switch to a different CV scheme variant (still leakage-safe) AND
  - increase ensemble diversity (different features/embeddings/propagation).

ZERO-OMISSION RULE
- You are not allowed to delete or shorten any requirement above.
- If any requirement conflicts with AGENTS.md, AGENTS.md wins, but you must explicitly document the conflict and how you resolved it in REPORT.md.

FINAL OUTPUT REQUIREMENTS (STRICT)
When you stop (ONLY after beating Top 1 public LB), you must output:
- The exact Top 1 target score you beat and your achieved public score.
- Kaggle submission ID / timestamp if available.
- The path(s) to the exact submission file(s) and a checksum for each.
- Repro commands that regenerate the winning submission exactly.
- The full file tree changes and a concise diff summary.

NOW EXECUTE.
- Immediately open AGENTS.md, extract requirements, then implement the full AUTOLOOP system and start running experiments and submissions until you beat Top 1.
