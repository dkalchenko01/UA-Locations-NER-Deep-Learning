import ast
import random
from collections import Counter

import numpy as np
import pandas as pd

_BAD = object()  # marker: loc_markers is present but cannot be parsed


def parse_spans(raw):
    """Parse a loc_markers cell once.

    Returns None if the cell is missing, _BAD if it cannot be parsed,
    otherwise the parsed object (normally a list of (start, end) tuples).
    """
    if not isinstance(raw, str):
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError):
        return _BAD


def has_pair_shape(spans):
    """True if spans is a list/tuple of (int, int) pairs."""
    return isinstance(spans, (list, tuple)) and all(
        isinstance(s, (list, tuple)) and len(s) == 2
        and all(isinstance(v, int) for v in s)
        for s in spans
    )


def scan_dataset(file_path, include_documents=False, check_uk_limit=False,
                 chunksize=100_000, text_sample_size=50_000, span_sample_size=100):
    """Single pass over a labeled CSV. Each loc_markers cell is parsed once and
    all span-level statistics are collected in one loop over the spans.

    Random streams (np.random.default_rng(42), random.Random(42)), chunk size and
    row order are the same as in the previous version, so samples are identical.
    """
    total = with_locations = duplicates = 0
    loc_counts, locations, audit, span_lengths = Counter(), Counter(), Counter(), Counter()
    audit_examples = {"overlap": [], "whitespace": [], "longest": []}
    total_spans = invalid_spans = 0
    invalid_examples, whitespace_examples = [], []
    text_sample, text_seen, text_rng = [], 0, np.random.default_rng(42)
    span_sample, span_seen, span_rng = [], 0, random.Random(42)
    limit_total = at_255 = over_255 = max_length = 0
    doc_counts = Counter()
    columns = ["text", "loc_markers"] + (["doc_id"] if include_documents else [])

    for chunk in pd.read_csv(file_path, usecols=columns, chunksize=chunksize):
        total += len(chunk)
        with_locations += int(
            chunk["loc_markers"].fillna("[]").str.strip().ne("[]").sum())
        duplicates += int(chunk["text"].duplicated().sum())

        # --- text lengths (vectorised stats + reservoir) ---
        sizes = chunk["text"].dropna().str.len()
        if check_uk_limit:
            limit_total += len(sizes)
            at_255 += int((sizes == 255).sum())
            over_255 += int((sizes > 255).sum())
            max_length = max(max_length, int(sizes.max()))
        for length in sizes.to_numpy():
            text_seen += 1
            if len(text_sample) < text_sample_size:
                text_sample.append(length)
            else:
                j = text_rng.integers(text_seen)
                if j < text_sample_size:
                    text_sample[j] = length

        # --- spans: one parse per row, one loop per row ---
        for text, raw in zip(chunk["text"], chunk["loc_markers"]):
            spans = parse_spans(raw)
            loc_counts[len(spans) if isinstance(spans, (list, tuple)) else 0] += 1

            if not isinstance(text, str) or spans is None:
                audit["missing"] += 1
                continue
            if spans is _BAD or not has_pair_shape(spans):
                audit["malformed"] += 1
                continue

            n = len(text)
            row_ok = all(0 <= s < e <= n for s, e in spans)
            if row_ok:
                audit["texts"] += 1
                ordered = sorted(spans)
                if any(ordered[i][0] < ordered[i - 1][1] for i in range(1, len(ordered))):
                    audit["overlap_texts"] += 1
                    if len(audit_examples["overlap"]) < 3:
                        audit_examples["overlap"].append((text[:150], ordered[:10]))
            else:
                audit["malformed"] += 1

            for start, end in spans:  # original order: samples/examples stay reproducible
                location = text[start:end]
                total_spans += 1
                locations[location] += 1

                if not (0 <= start < end <= n):
                    invalid_spans += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append((text, start, end))

                span_seen += 1
                record = {"text": text, "start": start, "end": end, "location": location}
                if len(span_sample) < span_sample_size:
                    span_sample.append(record)
                else:
                    j = span_rng.randrange(span_seen)
                    if j < span_sample_size:
                        span_sample[j] = record

                has_edge_space = bool(location) and (
                    location[0].isspace() or location[-1].isspace())
                if has_edge_space and len(whitespace_examples) < 5:
                    whitespace_examples.append({
                        "context": text[max(0, start - 40):start]
                        + "[" + location + "]" + text[end:end + 40],
                        "location": location, "start": start, "end": end,
                    })

                if row_ok:  # strict audit only on well-formed rows
                    audit["spans"] += 1
                    span_lengths[len(location)] += 1
                    if has_edge_space:
                        audit["whitespace_spans"] += 1
                        if len(audit_examples["whitespace"]) < 3:
                            audit_examples["whitespace"].append(repr(location))
                    if len(location) > 100:
                        audit["long_spans"] += 1
                        if len(audit_examples["longest"]) < 3:
                            audit_examples["longest"].append(location[:150])

        if include_documents:
            doc_counts.update(chunk["doc_id"].dropna().value_counts().to_dict())

    return {
        "coverage": {
            "dataset": file_path.name, "total_texts": total,
            "with_locations": with_locations, "without_locations": total - with_locations,
            "location_pct": round(with_locations / total * 100, 2),
        },
        "text_lengths": pd.Series(text_sample),
        "uk_limit": (limit_total, max_length, at_255, over_255),
        "loc_counts": loc_counts, "locations": locations,
        "offsets": (total_spans, invalid_spans, invalid_examples),
        "audit": audit, "span_lengths": span_lengths, "audit_examples": audit_examples,
        "whitespace_examples": whitespace_examples, "span_sample": pd.DataFrame(span_sample),
        "duplicates": {
            "total_rows": total, "duplicates_within_chunks": duplicates,
            "duplicates_pct_lower_bound": round(100 * duplicates / total, 2),
        },
        "doc_counts": doc_counts,
    }
