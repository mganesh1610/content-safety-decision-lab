# Data and method

The primary source is the authors' [ICWSM 2017 dataset](https://github.com/t-davidson/hate-speech-and-offensive-language), pinned to commit `cb50f7e9eb855a99abc51423bf40df0e26d6fe30`. Cite Davidson, Warmsley, Macy and Weber, *Automated Hate Speech Detection and the Problem of Offensive Language* (2017). Source license: MIT as provided upstream. This repository publishes aggregate results, not source posts.

## Schema

`labeled_data.csv`: index, count, hate_speech, offensive_language, neither, class, tweet. Class 0 is hate speech, 1 offensive language, 2 neither. The binary task is class 0 versus classes 1 and 2. Offensive language is deliberately retained as a hard negative; the two labels are not interchangeable. Only `tweet` enters the feature pipeline. Annotation vote columns are excluded to prevent target leakage.

HTML decoding, lowercasing, URL and mention placeholders, and whitespace normalization define a SHA-256 text group. Groups with conflicting original class labels are quarantined. One record per remaining normalized text is retained, with source-row multiplicity recorded. A deterministic stratified 60/20/20 train/validation/test split preserves original three-class composition. Duplicate text cannot cross partitions. Near duplicates, author independence, temporal transfer, and representativeness are unresolved.

TF-IDF is fitted exclusively on training text. Three regularization strengths are compared on validation PR-AUC. The selected training model is frozen; no train-plus-validation refit occurs. Class weighting means logistic scores are **not calibrated risk probabilities**. Threshold selection maximizes validation recall subject to review rate <=20% and false positive rate <=5%. Test outcomes do not influence selection.

Core measures: PR-AUC is average precision; ROC-AUC is pairwise discrimination; precision = TP/(TP+FP); recall = TP/(TP+FN); false positive rate = FP/(FP+TN); review rate = flagged/all. Wilson 95% intervals accompany the operating rates. Slice groups overlap, so their counts must not be summed. They describe URL presence, mention presence and text length, not demographic groups.

The local SQLite `evaluation` table contains `text_hash TEXT PRIMARY KEY, split TEXT, original_class INTEGER, target INTEGER, score REAL, flagged INTEGER`. Training rows have NULL scores and flags. SQL CTE metrics are reconciled independently to the frozen test confusion matrix. Data and database files are ignored by Git.

## Backup

[Google Civil Comments](https://huggingface.co/datasets/google/civil_comments) contains `text`, `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit` with official splits. Its CC0 toxicity annotations require a separate task, e.g. `toxicity >= 0.5`, its own preprocessing, deduplication and evaluation. This is a proposed backup adaptation, not an executed replication of the hate-speech task. Do not pool its labels with the primary source or treat it as independent evidence of current model accuracy.

The observed class balance is a sampling property. It cannot estimate live harm prevalence, exposure-weighted risk, policy violation rates, creator impact, or the harm reduction from moderation.
