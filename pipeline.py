"""
pipeline.py -- Athlytic NLP Pipeline
====================================
Analyzes Google Maps reviews of Gezira Sporting Club using core NLP techniques.

What this script does (step by step):
  1. Load Data          -- reads the scraped reviews JSON
  2. Text Preprocessing -- tokenize, remove stopwords, lemmatize, stem
  3. WordNet Explorer    -- explore word meanings, synonyms, hypernyms
  4. Competitor Detection-- find mentions of other clubs
  5. Syntax Analysis     -- NER and noun-adjective pair extraction (spaCy)
  6. Aspect Matching     -- keyword-based aspect detection
  7. Bigram Model        -- word association analysis
  8. VADER Sentiment     -- lexicon-based sentiment scoring
  9. Baseline ML         -- TF-IDF + SVM + Logistic Regression
 10. ABSA               -- aspect-based sentiment (VADER per sentence)
 11. Aggregate Analytics -- save final results to JSON
 12. Model Comparison    -- SVM vs LogReg accuracy/F1 table

Run:  python pipeline.py
"""

import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from langdetect import detect

# --- NLTK imports ------------------------------------------------------------
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk import pos_tag, bigrams, ConditionalFreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- sklearn imports ---------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# --- spaCy -------------------------------------------------------------------
import spacy


# =============================================================================
#  NLTK Data Downloads (run once, then cached)
# =============================================================================
print("Downloading NLTK data (if needed)...")
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet',
                 'averaged_perceptron_tagger_eng', 'vader_lexicon']:
    nltk.download(resource, quiet=True)
print("NLTK data ready.\n")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


# =============================================================================
#  STEP 1: Load Data
# =============================================================================
print("=" * 60)
print("  STEP 1: Load Data")
print("=" * 60)

with open("data/gezira_reviews.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)
with open("data/gezira_lowest500.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

# Combine both datasets
raw_data = data1 + data2
df = pd.DataFrame(raw_data)

# 1. Keep only reviews with meaningful text
df = df[df["text"].notna()]

# 2. Drop duplicates (just in case they share reviews)
df = df.drop_duplicates(subset=["text"])

# 3. Filter for English reviews only
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

print("  Filtering out non-English reviews...")
df = df[df["text"].apply(is_english)].reset_index(drop=True)
df["stars"] = df["stars"].astype(int)

print(f"  Total reviews loaded: {len(df)}")
print(f"\n  Star distribution:")
for star, count in df["stars"].value_counts().sort_index().items():
    print(f"    {star} stars: {count} reviews")
print()


# =============================================================================
#  STEP 2: Text Preprocessing (NLTK)
#  Covers: Text Pre-Processing, Computational Morphology
# =============================================================================
print("=" * 60)
print("  STEP 2: Text Preprocessing")
print("=" * 60)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
snowball = SnowballStemmer("english")


def nltk_pos_to_wordnet(tag):
    """Convert NLTK POS tag to WordNet POS tag for better lemmatization."""
    if tag.startswith("J"):
        return "a"  # adjective
    if tag.startswith("V"):
        return "v"  # verb
    if tag.startswith("N"):
        return "n"  # noun
    if tag.startswith("R"):
        return "r"  # adverb
    return "n"  # default to noun


def preprocess(text):
    """
    Full preprocessing pipeline for a single review:
      1. Remove non-ASCII characters (emojis, etc.)
      2. Normalize elongated words (e.g., 'sooo' -> 'so')
      3. Lowercase everything
      4. Tokenize into words
      5. Remove stopwords and non-alphabetic tokens
      6. POS tagging
      7. Lemmatize using WordNet (POS-aware)
      8. Stem using Porter and Snowball stemmers
    """
    # Clean the text
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)       # remove emojis / non-ASCII
    text = re.sub(r'(.)\1{2,}', r'\1', text)          # 'sooooo' -> 'so'
    text = text.lower().strip()

    # Tokenize
    tokens = word_tokenize(text)
    # Keep only alphabetic words that aren't stopwords
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    # POS tag the tokens
    tagged = pos_tag(tokens)

    # Lemmatize (uses POS tag for better results)
    lemmas = [lemmatizer.lemmatize(w, nltk_pos_to_wordnet(t)) for w, t in tagged]

    # Stem with both stemmers for comparison
    porter_stems = [porter.stem(w) for w in tokens]
    snowball_stems = [snowball.stem(w) for w in tokens]

    return {
        "tokens": tokens,
        "pos_tags": tagged,
        "lemmas": lemmas,
        "porter_stems": porter_stems,
        "snowball_stems": snowball_stems,
    }


# Apply preprocessing to all reviews
preprocessed = df["text"].apply(preprocess)
df["tokens"] = preprocessed.apply(lambda x: x["tokens"])
df["lemmas"] = preprocessed.apply(lambda x: x["lemmas"])
df["pos_tags"] = preprocessed.apply(lambda x: x["pos_tags"])
df["porter_stems"] = preprocessed.apply(lambda x: x["porter_stems"])
df["snowball_stems"] = preprocessed.apply(lambda x: x["snowball_stems"])

# Show results for the first review
print("\n  --- Sample: Review #0 ---")
print(f"  Original : {df['text'].iloc[0][:100]}...")
print(f"  Tokens   : {df['tokens'].iloc[0][:10]}")
print(f"  Lemmas   : {df['lemmas'].iloc[0][:10]}")
print(f"  POS Tags : {df['pos_tags'].iloc[0][:5]}")

# Morphology comparison table
print("\n  --- Morphology Comparison (first 5 tokens of Review #0) ---")
print(f"  {'Token':<15} {'Porter':<15} {'Snowball':<15} {'WordNet Lemma':<15}")
print(f"  {'-'*60}")
for i in range(min(5, len(df['tokens'].iloc[0]))):
    tok = df['tokens'].iloc[0][i]
    por = df['porter_stems'].iloc[0][i]
    sno = df['snowball_stems'].iloc[0][i]
    lem = df['lemmas'].iloc[0][i]
    print(f"  {tok:<15} {por:<15} {sno:<15} {lem:<15}")
print()


# =============================================================================
#  STEP 3: WordNet Exploration
#  Covers: Semantics -- using WordNet to understand word meanings
# =============================================================================
print("=" * 60)
print("  STEP 3: WordNet Exploration")
print("=" * 60)

# Pick some interesting words from the domain
wordnet_demo_words = ["pool", "clean", "staff", "expensive", "green", "facility"]

for word in wordnet_demo_words:
    synsets = wordnet.synsets(word)
    if not synsets:
        print(f"\n  '{word}': No synsets found")
        continue

    print(f"\n  '{word}' -- {len(synsets)} synset(s):")

    # Show the first 2 synsets with definition and examples
    for ss in synsets[:2]:
        print(f"    • {ss.name()} [{ss.pos()}]: {ss.definition()}")
        if ss.examples():
            print(f"      Example: \"{ss.examples()[0]}\"")

        # Hypernyms = more general concepts ("pool" is a kind of ...)
        hypernyms = ss.hypernyms()
        if hypernyms:
            print(f"      Hypernym (is-a): {hypernyms[0].name()} -- {hypernyms[0].definition()}")

        # Hyponyms = more specific concepts (kinds of "pool")
        hyponyms = ss.hyponyms()
        if hyponyms:
            hypo_names = [h.name().split('.')[0] for h in hyponyms[:3]]
            print(f"      Hyponyms (types): {', '.join(hypo_names)}")

    # Synonyms (lemma names from all synsets)
    all_lemmas = set()
    for ss in synsets:
        for lemma in ss.lemmas():
            if lemma.name() != word:
                all_lemmas.add(lemma.name().replace("_", " "))
    if all_lemmas:
        print(f"    Synonyms: {', '.join(list(all_lemmas)[:5])}")

# WordNet similarity demo
print("\n  --- Word Similarity (Wu-Palmer) ---")
word_pairs = [("pool", "facility"), ("clean", "dirty"), ("staff", "employee")]
for w1, w2 in word_pairs:
    s1 = wordnet.synsets(w1)
    s2 = wordnet.synsets(w2)
    if s1 and s2:
        sim = s1[0].wup_similarity(s2[0])
        if sim:
            print(f"  similarity('{w1}', '{w2}') = {sim:.3f}")
print()


# =============================================================================
#  STEP 4: Competitor Detection
# =============================================================================
print("=" * 60)
print("  STEP 4: Competitor Detection")
print("=" * 60)

COMPETITORS = [
    "gezira", "wadi degla", "ahly", "al ahly", "al-ahly", "zamalek",
    "shooting club", "maadi", "national club", "helio lido", "sporting",
    "smouha", "ittihad", "olympic", "heliopolis", "shams",
]


def find_competitors(text):
    """Find mentions of competing clubs using regex word boundaries."""
    text_lower = text.lower()
    return [c for c in COMPETITORS if re.search(r'\b' + re.escape(c) + r'\b', text_lower)]


df["competitor_mentions"] = df["text"].apply(find_competitors)

mentions = df[df["competitor_mentions"].apply(len) > 0]
print(f"  Reviews mentioning competitors: {len(mentions)}")
for _, row in mentions.head(3).iterrows():
    print(f"    [{', '.join(row['competitor_mentions'])}] {row['text'][:80]}...")
print()


# =============================================================================
#  STEP 5: Syntax Analysis -- NER & Noun-Adj Pairs (spaCy)
#  Covers: Syntax
# =============================================================================
print("=" * 60)
print("  STEP 5: Syntax Analysis (spaCy NER + Dependency Parsing)")
print("=" * 60)


def extract_syntax(text):
    """
    Use spaCy to extract:
      - Named Entities (NER): people, places, organizations, etc.
      - Noun-Adjective pairs: using dependency parsing to find
        which adjective describes which noun (e.g., "clean pool")
    """
    doc = nlp(text)

    # Extract named entities
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    # Extract noun-adjective pairs from the dependency tree
    noun_adj = []
    for token in doc:
        # If the token is an adjective and its head is a noun
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            noun_adj.append({"noun": token.head.text.lower(), "adj": token.text.lower()})
        # Also catch predicate adjectives (e.g., "the pool is clean")
        if token.dep_ in ("acomp", "attr") and token.pos_ == "ADJ":
            for child in token.head.children:
                if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                    noun_adj.append({"noun": child.text.lower(), "adj": token.text.lower()})

    return {"entities": entities, "noun_adj_pairs": noun_adj}


syntax_results = df["text"].apply(extract_syntax)
df["entities"] = syntax_results.apply(lambda x: x["entities"])
df["noun_adj_pairs"] = syntax_results.apply(lambda x: x["noun_adj_pairs"])

# Show sample results
print("\n  --- NER Results (Review #0) ---")
print(f"  Entities: {df['entities'].iloc[0]}")

print("\n  --- Noun-Adj Pairs (first 5 reviews) ---")
for i in range(min(5, len(df))):
    pairs = df["noun_adj_pairs"].iloc[i]
    if pairs:
        pair_strs = [f"{p['adj']} {p['noun']}" for p in pairs]
        print(f"    Review {i}: {', '.join(pair_strs)}")
print()


# =============================================================================
#  STEP 6: Aspect Keyword Matching
#  Covers: Semantics
# =============================================================================
print("=" * 60)
print("  STEP 6: Aspect Keyword Matching")
print("=" * 60)

# Each aspect has a list of keywords that signal it's being discussed
ASPECTS = {
    "Facilities": ["pool", "swimming", "lido", "gym", "fitness", "court", "tennis",
                   "squash", "football", "pitch", "track", "field", "shower", "sauna",
                   "spa", "golf", "equipment", "stadium"],
    "Prices": ["price", "fee", "fees", "expensive", "cheap", "affordable", "overpriced",
               "cost", "membership", "worth", "value", "money", "pricey", "reasonable"],
    "Community": ["community", "social", "atmosphere", "vibe", "people", "members",
                  "family", "friends", "friendly", "welcoming", "classy", "crowd"],
    "Greenery": ["green", "garden", "tree", "trees", "grass", "park", "nature",
                 "plants", "outdoor", "landscape", "lawn", "shade", "scenery", "nile"],
    "Service": ["staff", "service", "employee", "receptionist", "security", "guard",
                "trainer", "coach", "waiter", "helpful", "rude", "polite", "management"],
    "Crowd": ["crowded", "busy", "packed", "quiet", "empty", "rush", "peak",
              "weekend", "morning", "evening", "queue", "wait", "full"],
    "Food & Beverages": ["food", "restaurant", "cafe", "coffee", "drink", "menu",
                         "meal", "breakfast", "lunch", "dinner", "snack", "buffet"],
    "Parking": ["parking", "car", "spot", "garage", "valet", "traffic"],
    "Kids Area": ["kids", "children", "child", "playground", "play area", "toddler",
                  "slide", "swing"],
    "Cleanliness": ["clean", "dirty", "hygiene", "filthy", "messy", "smell",
                    "bathroom", "toilet", "trash", "maintenance", "maintained"],
}


def match_aspects(tokens):
    """Check which aspects a review mentions based on keyword matching."""
    token_set = set(tokens)
    text_joined = " ".join(tokens)
    matched = []
    for aspect, keywords in ASPECTS.items():
        for kw in keywords:
            if " " in kw:
                if kw in text_joined:
                    matched.append(aspect)
                    break
            elif kw in token_set:
                matched.append(aspect)
                break
    return matched


df["aspects"] = df["lemmas"].apply(match_aspects)

# Count how many times each aspect appears
all_aspects = [a for aspects in df["aspects"] for a in aspects]
aspect_counts = Counter(all_aspects).most_common()

print("\n  Aspect Mention Counts:")
for aspect, count in aspect_counts:
    bar = "#" * (count // 5)  # simple visual bar
    print(f"    {aspect:20s}: {count:4d}  {bar}")
print(f"\n  Reviews with no aspect detected: {sum(1 for a in df['aspects'] if not a)}")
print()


# =============================================================================
#  STEP 7: Bigram Language Model (NLTK)
#  Covers: Language Models
# =============================================================================
print("=" * 60)
print("  STEP 7: Bigram Language Model")
print("=" * 60)

# Build a conditional frequency distribution from all lemmas
all_lemmas = [lemma for lemma_list in df["lemmas"] for lemma in lemma_list]
cfd = ConditionalFreqDist(bigrams(all_lemmas))

# For each seed word, show what words most commonly follow it
seed_words = ["pool", "staff", "parking", "food", "green", "kid", "clean", "price", "time", "place", "court", "service", "member"]

print("\n  Word Associations (what word comes next?):")
for word in seed_words:
    if word in cfd:
        top = cfd[word].most_common(10)
        following = ", ".join([f"'{w}' ({c})" for w, c in top])
        print(f"    '{word}' -> {following}")
print()


# =============================================================================
#  STEP 8: VADER Sentiment Analysis (NLTK)
#  Covers: Semantics -- lexicon-based sentiment
# =============================================================================
print("=" * 60)
print("  STEP 8: VADER Sentiment Analysis")
print("=" * 60)

sia = SentimentIntensityAnalyzer()


def vader_sentiment(text):
    """Classify text as positive or negative using VADER compound score."""
    scores = sia.polarity_scores(text)
    return "positive" if scores["compound"] >= 0.05 else "negative"


df["vader_sentiment"] = df["text"].apply(vader_sentiment)

# Create labels from star ratings for comparison
df["star_label"] = df["stars"].apply(lambda s: "positive" if s >= 4 else "negative")

print("\n  VADER Sentiment Distribution:")
for label, count in df["vader_sentiment"].value_counts().items():
    print(f"    {label}: {count}")

print(f"\n  Star-based Label Distribution:")
for label, count in df["star_label"].value_counts().items():
    print(f"    {label}: {count}")

# How often do VADER and star ratings agree?
agreement = (df["vader_sentiment"] == df["star_label"]).mean()
print(f"\n  VADER vs Star Rating Agreement: {agreement:.1%}")

# Sample predictions
print("\n  Sample Predictions:")
for i in [0, 10, 20]:
    if i < len(df):
        print(f"    Stars={df['stars'].iloc[i]}  VADER={df['vader_sentiment'].iloc[i]}  "
              f"| {df['text'].iloc[i][:70]}...")
print()


# =============================================================================
#  STEP 9: Baseline ML -- TF-IDF + SVM / Logistic Regression
#  Covers: NLP with Machine Learning
# =============================================================================
print("=" * 60)
print("  STEP 9: Baseline ML (TF-IDF + SVM + LogReg)")
print("=" * 60)

# Skip 3-star reviews (they're ambiguous -- not clearly positive or negative)
ml_df = df[df["stars"] != 3].copy()
ml_df["label"] = ml_df["stars"].apply(lambda s: 1 if s >= 4 else 0)

# Use lemmatized text as input features
X_text = ml_df["lemmas"].apply(lambda l: " ".join(l))
y = ml_df["label"]

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text to TF-IDF features (max 3000 features)
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train SVM
svm = LinearSVC(random_state=42, max_iter=5000)
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)

# Train Logistic Regression
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_tfidf, y_train)
logreg_pred = logreg.predict(X_test_tfidf)

# Print classification reports
print("\n  --- SVM Classification Report ---")
print(classification_report(y_test, svm_pred, target_names=["negative", "positive"]))
print("  --- Logistic Regression Classification Report ---")
print(classification_report(y_test, logreg_pred, target_names=["negative", "positive"]))

# Save confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, name in [(axes[0], svm_pred, "SVM"), (axes[1], logreg_pred, "LogReg")]:
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["neg", "pos"])
    ax.set_yticklabels(["neg", "pos"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
print("  Saved confusion_matrix.png")
print()


# =============================================================================
#  STEP 10: Aspect-Based Sentiment Analysis (ABSA)
#  Covers: Advanced Semantic Analysis
# =============================================================================
print("=" * 60)
print("  STEP 10: Aspect-Based Sentiment (VADER per aspect)")
print("=" * 60)


def aspect_sentiment_vader(text, aspects_matched):
    """
    For each aspect mentioned in the review:
      1. Find the specific sentences that mention the aspect's keywords
      2. Run VADER on those sentences only
      3. Classify that aspect as positive or negative

    This way, a review like "The pool is great but the parking is awful"
    gets: Facilities=positive, Parking=negative
    """
    results = {}
    sentences = sent_tokenize(text)

    for aspect in aspects_matched:
        keywords = ASPECTS.get(aspect, [])
        # Find sentences mentioning this aspect
        relevant = [s for s in sentences if any(kw in s.lower() for kw in keywords)]

        if not relevant:
            results[aspect] = "unknown"
            continue

        # Analyze sentiment of the relevant sentences only
        aspect_text = " ".join(relevant)
        score = sia.polarity_scores(aspect_text)["compound"]
        results[aspect] = "positive" if score >= 0.05 else "negative"

    return results


# Run ABSA on reviews that have at least one aspect
sample = df[df["aspects"].apply(len) > 0].copy()
sample["absa_results"] = sample.apply(
    lambda r: aspect_sentiment_vader(r["text"], r["aspects"]), axis=1
)

print(f"\n  Analyzed {len(sample)} reviews with aspect mentions")
print("\n  Sample ABSA Results:")
for _, row in sample.head(5).iterrows():
    print(f"    [{row['stars']}*] {row['text'][:70]}...")
    for aspect, sentiment in row['absa_results'].items():
        marker = "[+]" if sentiment == "positive" else "[-]"
        print(f"         {marker} {aspect}: {sentiment}")
print()


# =============================================================================
#  STEP 11: Aggregate Analytics
# =============================================================================
print("=" * 60)
print("  STEP 11: Aggregate Analytics")
print("=" * 60)

aspect_analytics = {}
for aspect in ASPECTS:
    aspect_reviews = sample[sample["absa_results"].apply(lambda d: aspect in d)]
    if len(aspect_reviews) == 0:
        continue
    sentiments = aspect_reviews["absa_results"].apply(lambda d: d[aspect])
    pos = int((sentiments == "positive").sum())
    neg = int((sentiments == "negative").sum())
    total = pos + neg
    aspect_analytics[aspect] = {
        "mentions": total,
        "positive_pct": round(pos / total * 100, 1) if total else 0,
        "negative_pct": round(neg / total * 100, 1) if total else 0,
    }

print("\n  Aspect-Level Analytics:")
for aspect, stats in sorted(aspect_analytics.items(), key=lambda x: -x[1]["mentions"]):
    print(f"    {aspect:20s}: {stats['mentions']:3d} mentions | "
          f"+{stats['positive_pct']}% / -{stats['negative_pct']}%")

# Save to JSON
analytics_output = {
    "club": "Gezira Sporting Club",
    "total_reviews_analyzed": int(len(sample)),
    "aspect_analytics": aspect_analytics,
}
with open("gezira_analytics.json", "w") as f:
    json.dump(analytics_output, f, indent=2)
print("\n  Saved gezira_analytics.json")
print()


# =============================================================================
#  STEP 12: Model Comparison Summary
# =============================================================================
print("=" * 60)
print("  STEP 12: Model Comparison")
print("=" * 60)

svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
lr_acc = accuracy_score(y_test, logreg_pred)
lr_f1 = f1_score(y_test, logreg_pred)

print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1 Score':>10}")
print(f"  {'-'*45}")
print(f"  {'TF-IDF + SVM':<25} {svm_acc:>10.3f} {svm_f1:>10.3f}")
print(f"  {'TF-IDF + LogReg':<25} {lr_acc:>10.3f} {lr_f1:>10.3f}")

print(f"""
  Note: Both models predict overall review sentiment (pos/neg).
  The ABSA approach (Step 10) is better because it gives per-ASPECT
  sentiment -- e.g., a single review can be positive about Facilities
  but negative about Parking.
""")

print("=" * 60)
print("  PIPELINE COMPLETE!")
print("  Run 'streamlit run dashboard.py' to explore results visually.")
print("=" * 60)
