from src.eda_scan import parse_spans

def drop_conflicting_annotations(df):
    conflicts = df.groupby("text")["loc_markers"].transform("nunique") > 1
    return df.loc[~conflicts].copy()


def drop_duplicate_texts(df):
    normalized = df["text"].str.lower().str.split().str.join(" ")
    return df.loc[~normalized.duplicated()].copy()


def preprocess_data(df):
    df = drop_conflicting_annotations(df)
    df = drop_duplicate_texts(df)
    return df


def spans_to_bio(offsets, spans):
    labels = ["O"] * len(offsets)
    for start, end in spans:
        indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not indices:
            raise ValueError(f"No tokens for span {(start, end)}")
        if offsets[indices[0]][0] != start or offsets[indices[-1]][1] != end:
            raise ValueError(f"Partial alignment for span {(start, end)}")
        labels[indices[0]] = "B-LOC"
        for i in indices[1:]:
            labels[i] = "I-LOC"
    return labels


def prepare_bio_data(tokenizer, df, source):
    records = []
    skipped_ids = []
    for row_id, row in df.iterrows():
        spans = parse_spans(row["loc_markers"])
        encoding = tokenizer.encode(row["text"], add_special_tokens=False)
        try:
            labels = spans_to_bio(encoding.offsets, spans)
        except ValueError:
            skipped_ids.append(f"{source}_{row_id}")
            continue
        records.append({
            "text_id": f"{source}_{row_id}",
            "input_ids": encoding.ids,
            "offsets": encoding.offsets,
            "labels": labels
        })
    return records, skipped_ids


def bio_to_spans(labels, offsets):
    if len(labels) != len(offsets):
        raise ValueError("Labels and offsets must have equal lengths")
    spans = []
    start, end = None, None
    for label, (a, b) in zip(labels, offsets):
        if label == "B-LOC" or (label == "I-LOC" and start is None):
            if start is not None:
                spans.append((start, end))
            start, end = a, b
        elif label == "I-LOC":
            end = max(end, b)
        elif label == "O":
            if start is not None:
                spans.append((start, end))
                start, end = None, None
        else:
            raise ValueError(f"Unknown BIO label: {label}")
    if start is not None:
        spans.append((start, end))
    return spans
