# Ukrainian Location Extraction

## Data insights

**Notebook:** `01_eda.ipynb`

- Training data: 1.01M Ukrainian and 8.03M russian texts. Only 23% / 31% contain locations.
- Test: 477 posts, roughly 90% Ukrainian (character-based estimate). Median length is 264 characters, compared to 91–110 in training. **85%** texts have line breaks and **52%** match the emoji pattern.
- Ukrainian training texts have maximum **255 characters**, while test reaches 3,682. Models must handle longer posts and unfamiliar symbols; sentence-validation scores may overestimate test performance.
- Labels use character spans `[start, end)`. Some names have incorrect boundaries or irrelevant labels (despite the valid offsets).

## Validation

**Notebook:** `02_validation.ipynb`

We used GroupShuffleSplit to create one fixed validation set. Near-duplicate Ukrainian texts stay in one split; russian sentences are grouped by `doc_id`. All models use the same split for comparison. 

- Removed duplicates and texts with conflicting stored annotations. 
- Shared split: **seed 17**, Ukrainian **80/20**. To reflect the test language mix (mostly Ukrainian), we subsample russian training data to only **100,000** texts.

| Source | Train | Validation |
|---|---:|---:|
| Ukrainian | 805,983 | 201,498 |
| russian | 100,000 | 49,369 |

- No shared groups between splits; text lengths and location frequencies are similar.
- For addiitonal diagnostics we also combined 5,000 validation sentences into 1,000 noisy posts.
- We did not use the 26-post `labeling_sample.csv` for model selection or error diagnostics, because some location annotations are missed there.
- Main score: **official normalized micro F1**; repeated names count once per text. Additional metrics chosen for tracking: precision, recall, exact entity-level F1 and mean per-text F1.

## Preprocessing

**Notebook:** `03_preprocessing.ipynb`

Trained Whitespace and ByteLevel BPE on training text only, with **32,000 tokens** each. Then converted location spans into **BIO labels**. Labels can cover several subword tokens.
- ByteLevel had better coverage: 0% unknown test tokens, compared to 2.01% for Whitespace
- Checked that BIO labels recover the original spans: zero mismatches among aligned examples.
- Preserved original text and character offsets so predictions can be returned as exact text substrings, including original spelling and punctuation.

## Stacked BiGRU

**Notebook:** `04_bigru_stacked.ipynb`

We implemented a **3-layer bidirectional GRU**, reusing the existing training and evaluation pipeline. The initial hyperparameters were chosen to approximately match the best Stacked BiLSTM configuration (further tuning planned).

- **Tokenizer:** Whitespace BPE, vocabulary size 32,000.
- **Architecture:** trainable 128-dimensional embeddings -> 3 bidirectional GRU layers (hidden size 256 per direction) -> dropout 0.2 -> linear classifier predicting O/B-LOC/I-LOC.
- **Training:** Adam optimizer, learning rate 0.001, batch size 128, seed 17, trained for 3 epochs.
- **Inference:** full-message prediction, followed by BIO-constrained Viterbi decoding with line-break constraints (`viterbi_lines`).
- **Postprocessing:** word-boundary expansion, edge trimming, `clean_v3`, removal of predictions with empty locations.

### Results

We evaluated the checkpoints using the normalized micro F1 metric. The best result (after Viterbi decoding and postprocessing) was at **epoch 3**.

| Validation metric | F1 |
|---|---:|
| Ukrainian | 0.87600 |
| russian | 0.90177 |
| Combined | 0.88253 |

**Kaggle Public F1: 0.59084**.
