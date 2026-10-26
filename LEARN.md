# LEARN.md — what each file does and why (for the oral exam)

## Added overnight 2026-10-25/26 (by Claude, check every point)

### `scripts/run_rf.py` — random forest baseline (RQ1)
Reads the hand-crafted features from `data/features.parquet`, keeps the 28 columns of the `main` feature set, and trains a random forest on the training cells. It tries 12 configurations on validation with seed 0, picks the one with the highest validation macro-F1, then retrains that one with seeds 0, 1 and 2 for 4 and 7 classes. Every run is one row in `results/experiments.csv`; final runs also save their validation predictions.
- **Why `main` and not all features:** the raw darkfield intensity levels track the image background, which differs by phase, so the forest could learn the background instead of the cell (PROJECT.md decision 2026-10-24).
- **Why the imputer is inside a pipeline:** `SimpleImputer` is fitted inside `.fit()` on the training rows only, so validation cells never influence the medians (no leakage).
- **Why vary `min_samples_leaf` instead of `max_depth`:** both limit how detailed the trees get; a minimum leaf size adapts to how many cells reach a branch, which matters with a 300-cell class next to a 10,000-cell class.
- **Why `class_weight`:** without it the forest mostly predicts G1/S/G2; "balanced" gives each class equal total weight, "balanced_subsample" recomputes this inside each bootstrap sample.

### `scripts/fig_rf_importance.py` — which features the forest uses
Shuffles one feature column at a time in the validation set and measures how much the validation macro-F1 drops (permutation importance, 10 repeats). A big drop means the forest relies on that feature.
- **Why permutation importance and not the built-in (impurity) importance:** impurity importance is computed on training data and favours continuous features with many split points; permutation importance is measured on held-out cells with the metric we actually report.
- **Caveat:** correlated features share importance (area, perimeter and axis lengths all measure size), so a single feature can look less important than "size" as a whole really is.

### `src/models.py` — the CNNs
`SmallCNN`: four blocks of convolution → batch normalisation → ReLU → max pooling (66 → 33 → 16 → 8 → 4 px), global average pooling, dropout, one linear layer; 389,636 parameters. `ResNet18`: the standard 18-layer residual network from `timm`, with its first convolution changed from 3 to 2 input channels; 11.2 million parameters. Both have `embed(x)`, which returns the features just before the last layer (used for UMAP).
- **Why average the RGB filters into conv1:** ImageNet filters expect colour images. Our two channels are both grey-scale; the mean of the R, G and B filter is what the filter would compute on a grey image, so the pretrained edge/blob detectors are kept. Copying only R and G (another option) would be arbitrary.
- **Why upscale to 128 px for ResNet18:** ResNet18 shrinks the image by a factor of 32. A 66 px image would end at about 3 × 3 positions, and the pretrained filters were learnt on objects much larger than 66 px. 128 px gives 4 × 4 and objects closer to ImageNet scale, at about 2× the compute of 66 px (measured: 390 vs 780 images/s).
- **Why global average pooling:** it makes the classifier independent of the exact position of the cell and needs far fewer parameters than flattening.
- **Why batch normalisation:** keeps activations in a stable range so training from scratch works with larger learning rates.

### `src/train.py` — one training script for every CNN
Loads train and validation images (never test), normalises each channel with training statistics, trains with AdamW and a cosine learning-rate schedule, checks validation macro-F1 after every epoch, keeps the best weights and stops early. It logs one row per run, saves validation predictions and the best weights, and writes a checkpoint every epoch so a crashed run can resume.
- **Why normalise with training statistics only:** computing the mean/SD on validation cells too would let information from those cells into the model (a mild leak).
- **Why a weighted loss:** mitosis is 2% of cells; with plain cross-entropy the network can ignore it and still have low loss. Weighting class k by N / (K · n_k) makes every class contribute equally to the loss — the same idea as `class_weight="balanced"` in the forest.
- **Why 90° rotations and flips as augmentation:** a cell in a flow cytometer has no "up"; rotating by 90° and flipping move pixels exactly (no interpolation blur), so the network learns that orientation does not matter. Intensity jitter mimics small brightness differences between cells/runs.
- **Why early stopping on validation macro-F1 (not loss):** macro-F1 is the metric we report and select on; loss can keep falling while the rare class gets worse.
- **Why AdamW + cosine:** AdamW is Adam with weight decay applied correctly (decoupled from the gradient); cosine slowly lowers the learning rate to zero so the last epochs fine-tune instead of jumping around.
- **Why the data stay on the GPU as uint8:** 22,586 × 2 × 66 × 66 bytes ≈ 200 MB; converting per batch is fast and avoids a slow data loader on a laptop.

### `src/experiments.py` (changed)
Three columns added: `run_id` (a unique name per run, so a queue can skip finished runs), `device` and `git_commit` (which code version produced the number; "-dirty" means uncommitted changes).

### `src/interpret.py` — looking inside the CNNs
`load_trained` rebuilds a finished run from its log row and best checkpoint; `embeddings` gives the penultimate features; `grad_cam` makes a heat map of where the network looked; `mask_fraction`/`border_fraction` turn a heat map into numbers.
- **How Grad-CAM works (5 steps):** (1) run the image forward and store the activations of the last convolutional layer; (2) backpropagate the score of the class of interest to those activations; (3) average each channel's gradient over the image → one weight per channel; (4) weighted sum of the channels, ReLU (keep only evidence *for* the class); (5) upsample to 66 × 66.
- **Why the last conv layer:** it has the most class-specific features while still keeping a (coarse) spatial map: 8 × 8 for SmallCNN, 4 × 4 for ResNet18 at 128 px — so ResNet maps are blurrier.
- **Why measure "heat inside the mask" against "mask area":** if the heat were spread evenly, the share inside the mask would equal the mask's share of the image. Clearly more means the network looks at the cell; about equal or less suggests background/border use.

### `scripts/fig_umap.py` — RQ4
UMAP of validation cells in three spaces (hand-crafted features, SmallCNN embedding, ResNet18 embedding), coloured by the 7 phases although the networks were trained on 4 classes. If the phases appear in cell-cycle order along a path, the continuous cycle is recoverable from features learnt for a discrete task.
- **Why standardise the hand-crafted features:** UMAP uses distances; without scaling, area (hundreds of pixels) would dominate GLCM values (0–1).

### `scripts/run_rf_raw_dark.py` — RQ5 side experiment for the forest
Retrains the selected random forest on `raw_dark` (the main features plus the raw darkfield mean and percentiles). The raw darkfield level depends on the image background, and the background differs by phase.
- **How to read it:** if adding background-dependent features raises the score, the forest can profit from a shortcut. Validation result: +0.015 macro-F1 (0.647 vs 0.632), a small but consistent gain over 3 seeds.

### `scripts/make_results_table.py`, `scripts/make_table1.py`, `scripts/fig_model_comparison.py`, `scripts/fig_confusion.py`
They read `results/experiments.csv` (and saved predictions) and write the tables and figures. Nothing is typed in by hand. The "final" runs of each model are identified by their run_id (`<model>_c<classes>_final_s<seed>`).
- **Why mean ± SD over seeds:** a single run can be lucky; the SD shows how much the result moves when only the random seed changes (initialisation, batch order, augmentation).
- **Why row-normalised confusion matrices:** each row then shows the recall of one true class and where its errors go. Raw counts would be dominated by G1.

### `scripts/queue_phase3.sh`, `scripts/queue_phase5.sh`, `scripts/best_cnn_config.py`
Shell scripts that run all training jobs one after another in the background, so the work continues even if the chat stops. Each run is skipped if its run_id is already logged, and resumes from its last checkpoint if it was interrupted. `best_cnn_config.py` reads the 12 tuning rows of a model family and prints the arguments of the best one.
- **Why the same 12-configuration grid for every CNN:** RQ1/RQ2 compare model families; if one family got more tuning it could win just because of that.
- **Why cap epochs (4 for tuning, 12 for final runs):** one laptop GPU, one night. The cap is identical for all families, so the comparison stays fair, but all CNN numbers are probably a bit below what longer training would give.

### Learning curve (`--label-fraction`, `scripts/fig_learning_curve.py`)
The model trains on a stratified 1%, 5% or 20% subset of the training cells (the validation set stays complete). One "epoch" always shows as many images as the full training set has, so the small subsets are repeated many times.
- **Why equal gradient steps:** with a fixed number of epochs, a 1% run would get 100× fewer weight updates, and the curve would mostly measure "too little training", not "too few labels".
- **Why stratified:** with 1% (226 cells) a random subset could contain almost no mitotic cells (about 5 are expected).

### `scripts/fig_ablation.py` — RQ3 (preliminary)
Six SmallCNN runs: augmentation none/basic × balancing none/weighted/oversample, seed 0 only.
- **Weighted loss vs oversampling:** both make rare classes count more. Weighting multiplies their loss; oversampling shows their images more often (with repeats). With augmentation, repeated mitotic images are at least rotated/flipped differently each time.

### `scripts/pseudotime.py` — RQ4, simplest pseudotime
Takes the first principal component (PC1) of each feature space as a 1-D "time" axis and measures with Spearman's ρ whether it orders the cells G1 → S → G2 → mitosis.
- **Why Spearman:** phase order is a ranking, not a number on a scale; Spearman only uses ranks.
- **Caveat to say in the oral:** the CNN was trained to separate G1, S, G2 and mitosis, so an embedding that orders those four is partly what we trained for. The interesting part is what it does with phases it was never told apart (pro/meta/ana/telo, all "mitosis") and whether the order is continuous within a class.

### Examiner questions (practice for the closed-book oral)
<details>
<summary>5 likely questions with model answers — try answering first, then open</summary>

**1. Why do you report macro-F1 and not accuracy?**
Validation has 2,150 G1 cells but only 107 mitotic cells (2%). A model that always says G1 already reaches 44% accuracy (2,150 / 4,840), and one that is good on G1/S/G2 but never predicts mitosis can reach high accuracy while being useless for mitosis. Macro-F1 computes the F1 score of each class separately and averages them with equal weight, so failing on mitosis costs a quarter of the score in the 4-class task. The "Chance (majority)" row shows this: always predicting G1 gives macro-F1 0.154 although its G1 recall is 1.0.

**2. How do you know the CNN does not just learn the image background?**
Three checks. (a) Grad-CAM: for G1, G2 and mitosis, a much larger share of the heat lies inside the cell mask than the mask's share of the image (e.g. prophase 0.64 of the heat in 0.17 of the area). (b) For S phase the heat is mostly in the border ring (0.44, about the ring's area share 0.43), which is a warning sign we report, not hide. (c) The random-forest side experiment: adding raw darkfield levels, which follow the background, raised macro-F1 by only 0.015. Grad-CAM is a coarse map (8 × 8 or 4 × 4 positions), so it is an indication, not proof.

**3. Why did you never look at the test set, and what is the validation set for?**
The validation set is used to choose hyperparameters and the best epoch. Every choice made on it makes the validation score slightly optimistic, because we picked what happened to work best on those cells. The test set is touched once, at the end, with everything already fixed, so its score is an honest estimate for new cells. If we tuned on the test set, it would become a second validation set and nothing would remain to give an unbiased estimate.

**4. Why does ImageNet pretraining help on grey-scale cell images that look nothing like ImageNet photos?**
The first layers of a network trained on ImageNet learn general detectors: edges, blobs, textures at different scales. Those are useful for any image, including cells. Starting from them, the network needs fewer labelled examples and fewer updates to find good features than starting from random weights. In our validation runs, the pretrained ResNet18 beat the same architecture from scratch under the same epoch cap and tuning grid. Part of the gap may shrink with longer training, because our cap (12 epochs) favours the model that starts closer to a good solution.

**5. Why three seeds, and what does the ± mean?**
Training involves randomness: initial weights (for scratch models), batch order and augmentation. Running with seeds 0, 1, 2 shows how much the score moves when only that randomness changes. The ± is the standard deviation over the three runs. If two models differ by less than about two of these SDs, we should not claim one is better. Three seeds is the minimum for a rough estimate, not a statistical test.
</details>

### `scripts/border_check.py` — RQ5 follow-up: does the CNN use the background? (added overnight, extra)
Changes only the outer 8-pixel ring of each validation image and counts how many predictions change. "Flattened" sets the ring to its own median brightness (keeps the level, removes edges/artefacts); "common" sets it to the same brightness for every cell (removes the level as well).
- **How to read it:** the cell is in the centre and is not touched, so a network that only looked at the cell would not change its answers. A drop in macro-F1 means it uses the background.
- **Weak point to admit in the oral:** the edited image has a sharp step at the edge of the ring, which never occurs in real images. A network can react to such an unusual image even if it never used the background. A cleaner test would be to paste the ring of a cell from another phase (background swap).
