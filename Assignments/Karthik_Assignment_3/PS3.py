# ============================================
# - Multinomial Naive Bayes
# - Log-space scoring + optional Laplace smoothing
# ============================================

import csv
import math
import re

# -------- tokenizer: lowercase + keep only letters/apostrophes --------
WORD_RE = re.compile(r"[a-z']+")


def tokenize(text):
    text = text.lower()
    tokens = WORD_RE.findall(text)
    return tokens

# -------- CSV reader: extracts label and text--------


def read_labeled_csv(path):
    data = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # normalize all header names to lowercase
        headers = [h.strip().lower() for h in reader.fieldnames]
        # find reasonable header matches
        label_key = None
        text_key = None
        for h in reader.fieldnames:
            h_low = h.strip().lower()
            if h_low in ("label", "class", "category", "sentiment", "y"):
                label_key = h
            if h_low in ("text", "review", "content", "message"):
                text_key = h
        if label_key is None or text_key is None:
            raise ValueError(
                f"CSV must have a label and text column. Found headers: {reader.fieldnames}"
            )

        for row in reader:
            label = (row[label_key] or "").strip()
            text = (row[text_key] or "").strip()
            if label and text:
                data.append((label, text))
    return data


# --- minimal English stopwords (short list) ---
STOPWORDS = set("""
a an the and or but if then else when while of to in on at by for from as is are was were be been being
this that these those here there it its it's i you he she we they them me my your his her our their
with without into out about over under up down
""".split())

# --- Vocab builder ---


def build_vocab(train_data, remove_stopwords=False, min_df=1):
    """
    Returns: set of tokens kept in the vocabulary based on document frequency.
    - remove_stopwords: if True, drop tokens in STOPWORDS
    - min_df: keep tokens that appear in at least min_df training documents
    """
    df = {}                # document frequency per token
    for _, text in train_data:
        toks = tokenize(text)
        uniq = set(toks)
        for w in uniq:
            if remove_stopwords and w in STOPWORDS:
                continue
            df[w] = df.get(w, 0) + 1

    vocab = set()
    for w, freq in df.items():
        if freq >= min_df:
            vocab.add(w)
    return vocab

# -------- training: build priors and word counts per class --------


def build_model(train_data, laplace=True, remove_stopwords=False, min_df=1):
    """
    Trains multinomial NB with:
    - optional Laplace smoothing
    - optional stopword removal (only if remove_stopwords=True)
    - optional min_df pruning (drop tokens with DF < min_df)
    """
    # 1) build vocab from training (no leakage)
    vocab = build_vocab(
        train_data, remove_stopwords=remove_stopwords, min_df=min_df)

    classes = []
    class_counts = {}
    word_counts = {}
    total_tokens = {}

    def ensure_class(c):
        if c not in class_counts:
            class_counts[c] = 0
            word_counts[c] = {}
            total_tokens[c] = 0
            classes.append(c)

    # 2) count words but ONLY keep tokens that survived in vocab
    for label, text in train_data:
        ensure_class(label)
        class_counts[label] += 1

        tokens = tokenize(text)
        for w in tokens:
            if w not in vocab:
                continue
            word_counts[label][w] = word_counts[label].get(w, 0) + 1
            total_tokens[label] += 1

    # 3) priors
    N = len(train_data)
    priors = {}
    for c in classes:
        priors[c] = class_counts[c] / N if N > 0 else 0.0

    model = {
        "classes": classes,
        "priors": priors,
        "word_counts": word_counts,
        "total_tokens": total_tokens,
        "vocab": vocab,
        "V": len(vocab),
        "laplace": laplace,
    }
    return model
# -------- log P(w|c) --------


def log_likelihood(w, c, model):
    wc = model["word_counts"][c].get(w, 0)
    T = model["total_tokens"][c]
    V = model["V"]

    if model["laplace"]:
        # Laplace smoothing: (count + 1) / (T + V)
        denom = T + V
        if denom == 0:
            return float("-inf")
        return math.log((wc + 1) / denom)
    else:
        if wc == 0 or T == 0:
            return float("-inf")
        return math.log(wc / T)

# -------- predict probabilities for one text --------


def predict_proba(text, model):
    tokens = tokenize(text)

    # term frequency (counts in this doc), ignoring tokens not in vocab
    tf = {}
    for w in tokens:
        if w in model["vocab"]:
            tf[w] = tf.get(w, 0) + 1

    # log-scores per class
    log_scores = {}
    for c in model["classes"]:
        # start with log prior
        prior = model["priors"][c]
        if prior <= 0.0:
            s = float("-inf")
        else:
            s = math.log(prior)

        # add token contributions: sum tf * log P(w|c)
        for w, f in tf.items():
            ll = log_likelihood(w, c, model)
            if ll == float("-inf"):
                # if unsmoothed and wc==0, skip (equivalent to multiply by 1 in linear space)
                continue
            s += f * ll
        log_scores[c] = s

    # normalize via log-sum-exp
    # (convert log-scores to probabilities that sum to 1)
    maxlog = None
    for c in model["classes"]:
        val = log_scores[c]
        if maxlog is None or val > maxlog:
            maxlog = val

    exps = {}
    for c in model["classes"]:
        exps[c] = math.exp(log_scores[c] - maxlog)
    Z = 0.0
    for c in model["classes"]:
        Z += exps[c]

    probs = {}
    if Z == 0.0:
        # fallback: uniform if something degenerated
        k = len(model["classes"])
        for c in model["classes"]:
            probs[c] = 1.0 / k if k > 0 else 0.0
    else:
        for c in model["classes"]:
            probs[c] = exps[c] / Z
    return probs


def predict(text, model):
    probs = predict_proba(text, model)
    # argmax by simple loop
    best_c = None
    best_p = -1.0
    for c in model["classes"]:
        p = probs[c]
        if p > best_p:
            best_p = p
            best_c = c
    return best_c

# -------- evaluation: accuracy, macro-precision, macro-recall, confusion --------


def evaluate(test_data, model):
    classes = model["classes"]
    index = {}
    for i, c in enumerate(classes):
        index[c] = i
    C = len(classes)

    # confusion: rows = actual, cols = predicted
    conf = []
    for _ in range(C):
        conf.append([0] * C)

    # fill confusion
    for y, text in test_data:
        yhat = predict(text, model)
        i = index[y]
        j = index[yhat]
        conf[i][j] += 1

    # accuracy
    total = 0
    correct = 0
    for i in range(C):
        for j in range(C):
            total += conf[i][j]
            if i == j:
                correct += conf[i][j]
    accuracy = (correct / total) if total > 0 else 0.0

    # per-class precision/recall
    precision = {}
    recall = {}
    for i, c in enumerate(classes):
        tp = conf[i][i]
        fp = 0
        fn = 0
        for r in range(C):
            if r != i:
                fp += conf[r][i]  # predicted i but actually r
                fn += conf[i][r]  # actually i but predicted r
        precision[c] = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall[c] = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # macro-averages
    macro_p = 0.0
    macro_r = 0.0
    for c in classes:
        macro_p += precision[c]
        macro_r += recall[c]
    if C > 0:
        macro_p /= C
        macro_r /= C

    return accuracy, macro_p, macro_r, conf

# -------- print confusion matrix--------


def print_confusion(conf, classes):
    print("Confusion matrix (rows=actual, cols=predicted):")
    header = [""] + classes
    print("\t".join(header))
    for i, row in enumerate(conf):
        cells = [classes[i]] + [str(x) for x in row]
        print("\t".join(cells))


# -------- main: runs both datasets, unsmoothed and smoothed --------
if __name__ == "__main__":
    # Your exact paths
    reviews_train = "/Users/karthikmattu/Documents/Code/IDAI610_PS3/Data/dataset_1_review/reviews_polarity_train.csv"
    reviews_test = "/Users/karthikmattu/Documents/Code/IDAI610_PS3/Data/dataset_1_review/reviews_polarity_test.csv"
    news_train = "/Users/karthikmattu/Documents/Code/IDAI610_PS3/Data/dataset_1_newsgroup/newsgroup_train.csv"
    news_test = "/Users/karthikmattu/Documents/Code/IDAI610_PS3/Data/dataset_1_newsgroup/newsgroup_test.csv"

    # --- Movie Reviews: unsmoothed ---
    tr_reviews = read_labeled_csv(reviews_train)
    te_reviews = read_labeled_csv(reviews_test)

    model_r_unsm = build_model(tr_reviews, laplace=False)
    acc, mp, mr, conf = evaluate(te_reviews, model_r_unsm)
    print("\n=== Movie Reviews (unsmoothed) ===")
    print("Classes:", model_r_unsm["classes"])
    print("Vocab size:", model_r_unsm["V"])
    print(
        f"Accuracy: {acc:.4f} | Macro-Precision: {mp:.4f} | Macro-Recall: {mr:.4f}")
    print_confusion(conf, model_r_unsm["classes"])

    # --- Movie Reviews: Laplace smoothed (Q7) ---
    model_r_sm = build_model(tr_reviews, laplace=True)
    acc, mp, mr, conf = evaluate(te_reviews, model_r_sm)
    print("\n=== Movie Reviews (Laplace smoothed) ===")
    print("Classes:", model_r_sm["classes"])
    print("Vocab size:", model_r_sm["V"])
    print(
        f"Accuracy: {acc:.4f} | Macro-Precision: {mp:.4f} | Macro-Recall: {mr:.4f}")
    print_confusion(conf, model_r_sm["classes"])
   # --- 20 Newsgroups subset: UNSMOOTHED (optional) + SMOOTHED with pruning/stopwords ---
    tr_news = read_labeled_csv(news_train)
    te_news = read_labeled_csv(news_test)

    # (optional) unsmoothed to show contrast; no pruning/stopwords
    model_news_unsm = build_model(
        tr_news, laplace=False, remove_stopwords=False, min_df=1)
    acc, mp, mr, conf = evaluate(te_news, model_news_unsm)
    print("\n=== 20 Newsgroups subset (UNSMOOTHED) ===")
    print("Classes:", model_news_unsm["classes"])
    print("Vocab size:", model_news_unsm["V"])
    print(
        f"Accuracy: {acc:.4f} | Macro-Precision: {mp:.4f} | Macro-Recall: {mr:.4f}")
    print_confusion(conf, model_news_unsm["classes"])

    # Laplace + min_df=2 + stopwords for 20NG
    model_news = build_model(tr_news, laplace=True,
                             remove_stopwords=True, min_df=2)
    acc, mp, mr, conf = evaluate(te_news, model_news)
    print("\n=== 20 Newsgroups subset (Laplace + min_df=2 + stopwords) ===")
    print("Classes:", model_news["classes"])
    print("Vocab size:", model_news["V"])
    print(
        f"Accuracy: {acc:.4f} | Macro-Precision: {mp:.4f} | Macro-Recall: {mr:.4f}")
    print_confusion(conf, model_news["classes"])
