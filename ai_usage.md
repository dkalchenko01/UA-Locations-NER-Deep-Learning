**Prompt:**

i want to compare how many annotated locations each text contains in the uk and ru datasets. can you plot the percentage of texts with 0, 1, 2, 3, and 4+ locations for both datasets on the same grouped bar chart? add percentage labels above the bars and make it readable

**Response:**

We can group the number of annotated locations into five categories (`0`, `1`, `2`, `3`, and `4+`), calculate their percentage in each dataset, and display them side by side.

```
fig, ax = plt.subplots(figsize=(9, 5))x = np.arange(5)width = 0.35for i, (label, result) in enumerate(scans.items()):    counts = result["loc_counts"]    pct = pd.Series(counts).groupby(        lambda n: min(n, 4)    ).sum().reindex(range(5), fill_value=0)    pct = pct / sum(counts.values()) * 100    bars = ax.bar(x + (i - 0.5) * width, pct, width, label=label.upper())    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)ax.set_xticks(x, ["0", "1", "2", "3", "4+"])ax.set_xlabel("Number of annotated locations per text")ax.set_ylabel("Percentage of texts (%)")ax.set_title("Distribution of Annotated Locations: UK vs RU")ax.legend()ax.set_ylim(0, 90)plt.tight_layout()plt.show()
```

The chart compares the distribution of annotated locations between the two datasets. It shows that most texts contain no annotated locations, while texts with multiple locations are less frequent.

Prompt:

help me check if location names mix latin and cyryllic characters in the uk and ru datasets. how to show the counts and top 5 examples for each type

Answer:

We can check whether location names contain characters from both Latin and Cyrillic alphabets using regular expressions.

The analysis distinguishes two types of mixed-alphabet cases:

1. `within_word` — Latin and Cyrillic characters occur in the same word. For example, a word could contain a Latin character in the middle of an otherwise Cyrillic spelling.
2. `cross_word` — Latin and Cyrillic characters occur in the same location name, but in separate words. For example, one word is written in Latin and another in Cyrillic.

First, we define regular expressions to detect characters from each alphabet:

```
LAT = re.compile(r"[A-Za-z]")CYR = re.compile(r"[\u0400-\u04FF]")
```

Next, we define a helper function `mixing_type(name)` that classifies a location name.

```
def mixing_type(name):    if any(LAT.search(w) and CYR.search(w) for w in name.split()):        return "within_word"    if LAT.search(name) and CYR.search(name):        return "cross_word"    return None
```

The function works in the following order:

- `name.split()` separates the location name into individual words.
- The first condition checks whether at least one individual word contains both Latin and Cyrillic characters. If so, the function returns `"within_word"`.
- If the first condition is not satisfied, the second condition checks whether the entire location name contains both alphabets. If it does, the function returns `"cross_word"`.
- If neither condition is satisfied, the function returns `None`, meaning that the location name is not mixed-alphabet.

The order is important: a location name containing both alphabets within one word is classified as `within_word`, rather than `cross_word`.

Then, we apply this function separately to each dataset in `scans`:

```
for label, result in scans.items():    locations = result["locations"]    total = sum(locations.values())    df = pd.DataFrame([(n, c, mixing_type(n)) for n, c in locations.items() if mixing_type(n)],        columns=["location", "count", "type"])    summary = df.groupby("type").agg(unique=("location", "size"), occurrences=("count", "sum"))    summary["share_of_occurrences_%"] = (summary["occurrences"] / total * 100).round(3)    print(label)    display(summary)    display(df.sort_values("count", ascending=False).groupby("type").head(5))
```

What this part does:

1. `locations = result["locations"]` retrieves the location-frequency data for the current dataset.
2. `total = sum(locations.values())` calculates the total number of location occurrences in that dataset. This is used as the denominator when calculating percentages.
3. The list comprehension iterates over every unique location name and its occurrence count:
    - `n` represents the location name.
    - `c` represents its frequency.
    - `mixing_type(n)` determines its category.
    - Names classified as `None` are excluded because we only want mixed-alphabet examples.
4. `pd.DataFrame(...)` creates a table with three columns: `location`, `count`, and `type`.
5. `groupby("type").agg(...)` groups the results into `within_word` and `cross_word` categories and calculates:
    - `unique` — the number of unique location names in each category.
    - `occurrences` — the sum of their occurrence counts.
6. `share_of_occurrences_%` calculates what percentage of all location occurrences in the dataset belongs to each mixed-alphabet category. The result is rounded to three decimal places.
7. Finally, the code sorts all mixed-alphabet location names by their occurrence counts in descending order, groups them by category, and displays the five most frequent examples from each category.

The loop repeats the same analysis for every dataset in `scans`, so the UK and RU results can be compared separately.

Expected output:

For each dataset, we get two outputs:

- A summary table showing the number of unique mixed-alphabet location names, their total occurrences, and their percentage of all location occurrences.
- A table showing the five most frequent location names in each mixed-alphabet category.

Why this analysis is useful for our NER task:

This is a data-quality and language-characteristics check. Since the project combines Ukrainian, Russian, and potentially mixed-language texts, location names may also contain inconsistent alphabet usage. For example, a name might be mostly Cyrillic but include a Latin character, or combine words written in different alphabets.

Identifying these cases helps us understand the actual contents of the dataset and detect potentially unusual spellings that could affect tokenization and location extraction.

It is important to note that this code only identifies and counts mixed-alphabet location names. It does not normalize, correct, or replace any characters in the original dataset.

Prompt:

I want to check for near-duplicate texts in the uk dataset. there are over a million texts, so comparing every pair is not rational at all. what methods can we use for this? discuss and then help me implement it step by step and then check how similar the texts actually are

Answer:

Since the dataset contains more than a million texts, comparing every possible pair would be computationally expensive. We can use MinHash + Locality Sensitive Hashing (LSH) to make the search more efficient.

The idea is to first find potentially similar texts using MinHash and LSH, and then calculate their actual Jaccard similarity to evaluate the candidate pairs.

We can split this into several steps.

1. Install and import the library

We will use `datasketch`, which provides implementations of MinHash and MinHashLSH.

```
%pip install -q datasketch
```

```
from datasketch import MinHash, MinHashLSH
```

2. Prepare text shingling and MinHash functions

First, we convert each text into a set of word-based shingles. Here, `k=3` means that each shingle consists of three consecutive words.

Then, we calculate a MinHash signature for each text using 128 permutations.

```
def shingles(text, k=3):    words = re.sub(r"\W+", " ", text.lower()).split()    return {" ".join(words[i:i+k]) for i in range(max(1, len(words)-k+1))}def minhash(text, k=3, num_perm=128):    m = MinHash(num_perm=num_perm)    for s in shingles(text, k):        m.update(s.encode("utf8"))    return m
```

3. Generate candidate pairs using MinHashLSH

We start with the UK texts, removing missing values and exact duplicates.

Then we initialize MinHashLSH with a similarity threshold of `0.7`. For every text, we:

- Generate its MinHash signature.
- Query LSH to find previously indexed texts that may be similar.
- Store the candidate pairs.
- Insert the current text into the index.

This way, we avoid comparing every text against every other text.

```
uk_texts = uk_df["text"].dropna().drop_duplicates()lsh_uk = MinHashLSH(threshold=0.7, num_perm=128)pairs_uk = set()for i, t in uk_texts.items():    h = minhash(t)    for j in lsh_uk.query(h):        pairs_uk.add((int(j), i))    lsh_uk.insert(str(i), h)print(f"Texts checked: {len(uk_texts):,}")print(f"Candidate pairs: {len(pairs_uk):,}")print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")
```

The output shown in the notebook:

```
Texts checked: 1,007,546
Candidate pairs: 6,356
Available RAM: 32.66 GB
```

This gives us a much smaller set of candidate pairs to investigate further.

4. Calculate actual Jaccard similarity

MinHashLSH is used to retrieve potentially similar pairs, but we still need to calculate their actual similarity.

For this, we use Jaccard similarity between the sets of word trigrams:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

A score closer to 1 means that the two texts share more shingles.

```
similarities_uk = []for i, j in pairs_uk:    A = shingles(uk_texts.loc[i], k=3)    B = shingles(uk_texts.loc[j], k=3)    score = len(A & B) / len(A | B)    similarities_uk.append((i, j, score))scores_uk = pd.DataFrame(similarities_uk, columns=["id1", "id2", "jaccard"])print(scores_uk["jaccard"].describe())print(f"Pairs with Jaccard >= 0.7: {(scores_uk['jaccard'] >= 0.7).sum():,}")
```

This final step gives us the actual Jaccard similarity distribution for the candidate pairs and tells us how many have a similarity of at least 0.7.

### What we are doing overall

1. Text preprocessing
    
    Convert each text into a set of consecutive three-word shingles.
    
2. MinHash
    
    Create a compact signature representing the shingle set.
    
3. LSH candidate search
    
    Find potentially similar texts without checking every possible pair.
    
4. Jaccard verification
    
    Calculate the actual similarity of the candidate pairs.
    

Important distinction: the LSH threshold of `0.7` is used to retrieve candidate pairs approximately. It does not guarantee that every pair with an actual Jaccard similarity above 0.7 will be found, nor that every retrieved pair will have a score above 0.7. That is why we calculate Jaccard similarity separately.

The code in the notebook performs this analysis for the UK dataset. The same approach can later be applied to the RU dataset.

Prompt:

Now I need to split the cleaned uk dataset into train and validation (80/20). since we already have near-duplicate groups I want to make sure similar texts don't end up in both train amd val sets. what is the best way to do this? how to check if the location distribution is similar in both sets

Answer:

Since we already identified near-duplicate groups, I wouldn't recommend using a regular random split. It could place very similar texts in both training and validation, which would cause data leakage and potentially make our validation results overly optimistic.

A better option here is `GroupShuffleSplit` from scikit-learn. It allows us to split the dataset while keeping all texts belonging to the same near-duplicate group together.

We'll use `near_dup_group_id` as the grouping variable and allocate approximately 80% of the texts to training and 20% to validation.

#### Cell 1. Split the dataset

```
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=17)
train_idx, val_idx = next(splitter.split(uk_clean, groups=uk_clean["near_dup_group_id"]))

uk_train = uk_clean.iloc[train_idx]
uk_val = uk_clean.iloc[val_idx]

print("Train:", len(uk_train), f"({len(uk_train) / len(uk_clean):.1%})")
print("Validation:", len(uk_val), f"({len(uk_val) / len(uk_clean):.1%})")
print("Shared groups:", len(set(uk_train["near_dup_group_id"]) & set(uk_val["near_dup_group_id"])))
```

Here:

- `n_splits=1` creates one train/validation split.
- `test_size=0.2` allocates approximately 20% of the groups to validation.
- `random_state=17` makes the split reproducible.
- `groups=uk_clean["near_dup_group_id"]` ensures that texts from the same group aren't split between train and validation.

The final check calculates the intersection of group IDs in both sets. Ideally, this should be zero.

#### Cell 2. Check the location balance

Now we should check whether the two subsets have similar distributions of annotated locations.

I suggest comparing two metrics:

- Location-positive rate: the proportion of texts containing at least one annotated location.
- Mean locations per text: the average number of annotated locations per text, including texts with zero locations.

Since `loc_markers` might contain either a string representation of a Python list or an actual list, we'll create a small helper function to handle both cases.

```
def count_locations(df):
    return df["loc_markers"].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x))

train_counts = count_locations(uk_train)
val_counts = count_locations(uk_val)

balance = pd.DataFrame({
    "Train": [(train_counts > 0).mean(), train_counts.mean()],
    "Validation": [(val_counts > 0).mean(), val_counts.mean()]
}, index=["Location-positive rate", "Mean locations per text"])

display(balance.round(4))
```

The `count_locations()` function calculates the number of annotated locations for each text.

Then:

- `(train_counts > 0).mean()` gives us the proportion of training texts with at least one location.
- `train_counts.mean()` calculates the average number of locations per training text.
- We repeat the same calculations for validation.
- Finally, we combine everything into a DataFrame so the distributions can be compared directly.

Prompt:
can we also check the distribution of the number of locations per text in train and val? and also I want to see how many texts contain multiword locations. the main goal is tp check whether the split is balanced

Answer:

Yes, this would be a useful additional check. We've already compared the location-positive rate and average number of locations per text, but those metrics don't show the full distribution.

For example, two datasets could have almost identical averages while containing different proportions of texts with multiple locations.

I suggest checking two things:

1. Location count distribution: what percentage of texts contains 0, 1, 2, 3, or 4+ annotated locations.
2. Multiword locations: what percentage of texts contains at least one location consisting of multiple words.

We can calculate both using one helper function.

#### Cell 1. Calculate location statistics

def location_stats(df):
markers = df["loc_markers"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
counts = markers.apply(len)
multiword = [
any(len(text[start:end].split()) > 1 for start, end in spans)
for text, spans in zip(df["text"], markers)
]
return counts.clip(upper=4).value_counts(normalize=True).sort_index(), sum(multiword) / len(df)

train_dist, train_multi = location_stats(uk_train)
val_dist, val_multi = location_stats(uk_val)

Here's what the function does.

First, it processes the `loc_markers` column. If the annotations are stored as strings, `ast.literal_eval()` converts them into Python objects. Otherwise, it uses them directly.

Then `markers.apply(len)` counts the number of annotated locations in each text.

For the distribution, we use:

```
counts.clip(upper=4).value_counts(normalize=True).sort_index()
```

This groups all texts containing four or more locations into one category and calculates the proportion of texts in each category.

For multiword locations, we use the annotation offsets:

```
any(len(text[start:end].split()) > 1 for start, end in spans)
```

For each annotated span, `text[start:end]` extracts the corresponding location from the original text. We then split it into words and check whether it contains more than one word.

The `any()` function returns `True` if at least one location in that text is multiword.

Finally, dividing the number of such texts by the total number of texts gives us the proportion containing at least one multiword location.

#### Cell 2. Compare location count distributions

Now we can combine the train and validation distributions into a single DataFrame.

```
distribution = pd.DataFrame({    "Train": train_dist,    "Validation": val_dist}).fillna(0).rename(index={4: "4+"})display((distribution * 100).round(2))
```

Here, `fillna(0)` handles categories that might be missing from one of the subsets, and `rename(index={4: "4+"})` makes it clear that the final category includes all texts with four or more locations.

Multiplying by 100 converts the proportions into percentages.

The output will let us compare how common each location-count category is in train and validation.

Ideally, the percentages should be reasonably similar across both sets. Large differences could indicate that one subset contains substantially more complex examples with multiple annotated locations.

#### Cell 3. Compare multiword location frequencies

Finally, let's display the percentage of texts containing at least one multiword location.

```
display(pd.DataFrame({    "Train": [train_multi * 100],    "Validation": [val_multi * 100]}, index=["Texts with multiword locations (%)"]).round(2))
```

This gives us a separate comparison of multiword entities.

Multiword locations are particularly relevant for our NER task because the model needs to identify the entire entity rather than individual words.

For example, a location such as `Київська область` should be recognized as one complete location span.

If validation contains significantly fewer multiword locations than training, it might not adequately reflect some of the entity-boundary challenges the model needs to handle.

### What we can conclude

These checks complement our previous train/validation analysis.

We now have several ways to evaluate whether the subsets are comparable: location-positive rate, mean number of locations per text, distribution of location counts, and percentage of texts with multiword locations.

If the new distributions are also similar, that would provide additional evidence that our group-based split is reasonably balanced in terms of location annotations.

However, we should still distinguish between annotation balance and complete dataset similarity. These checks don't tell us whether the two subsets have identical text lengths, vocabulary, or linguistic characteristics.

Also, this multiword check is based on whitespace separation. It identifies locations containing multiple whitespace-separated words, but it won't classify a hyphenated name such as `Івано-Франківськ` as multiword.

For now, we can use these statistics to evaluate whether validation contains a comparable range of simple and more complex annotated location examples, without modifying the split itself.






