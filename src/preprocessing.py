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