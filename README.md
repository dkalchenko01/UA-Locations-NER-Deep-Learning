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
