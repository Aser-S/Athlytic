# Athlytic — Project Report
## NLP Aspect-Based Sentiment Analysis for Sporting Club Reviews

**Course:** CSE389 — Natural Language Processing  
**Dataset:** Gezira Sporting Club, Google Maps Reviews

---

## 1. Introduction

Athlytic is an NLP pipeline that analyzes Google Maps reviews of Gezira Sporting Club to extract **per-aspect sentiment** — understanding not just whether a review is positive or negative, but what *specific aspects* (facilities, parking, service, etc.) people are praising or complaining about.

This goes beyond standard sentiment analysis: a single review saying *"The pool is beautiful, but the parking is awful"* gets decomposed into **Facilities: positive** and **Parking: negative**.

---

## 2. Database Selection

### Why Google Maps Reviews?

We chose Google Maps reviews because:
- **Real-world data**: Reviews are written by actual club members and visitors
- **Star ratings**: Each review has a 1-5 star rating we can use as ground truth labels
- **Rich text**: Reviews discuss multiple aspects (facilities, pricing, staff, cleanliness)
- **English content**: Sufficient English reviews available for NLP processing

### The Club: Gezira Sporting Club

Gezira Sporting Club is one of Egypt's oldest and most prominent sporting clubs, located on Zamalek island in Cairo. It was chosen because:
- High volume of English reviews (500+)
- Reviews cover diverse aspects (pool, gym, restaurants, parking, greenery)
- Active membership base with varied opinions

---

## 3. Data Collection

### How We Scraped the Data

We used an **online Google Maps scraper tool** to extract reviews:

1. **Input**: The Google Maps URL for Gezira Sporting Club
2. **Tool**: An online scraping service that extracts Google Maps reviews without needing an API key
3. **Output**: Exported as `Gezira-Scraper.json` — a JSON file containing all review data

Each review record contains:
- `name` — reviewer name
- `text` — the review text
- `stars` — rating (1-5)
- `publishAt` — date of the review

### Dataset Statistics

| Metric | Value |
|---|---|
| Total reviews scraped | ~500 |
| Reviews with text (>10 chars) | ~494 |
| Language | English |
| Star distribution | ~88% are 4-5 stars |

---

## 4. Pipeline Architecture

```
Gezira-Scraper.json (500 reviews)
  │
  ├── Step 1:  Load Data (JSON → DataFrame)
  ├── Step 2:  Text Preprocessing (NLTK: tokenize, stopwords, POS-tag, lemmatize, stem)
  ├── Step 3:  WordNet Exploration (synsets, definitions, hypernyms, similarity)
  ├── Step 4:  Competitor Detection (regex word-boundary matching)
  ├── Step 5:  Syntax Analysis (spaCy: NER + dependency parsing → noun-adj pairs)
  ├── Step 6:  Aspect Keyword Matching (dictionary-based, 10 aspects)
  ├── Step 7:  Bigram Language Model (NLTK ConditionalFreqDist)
  ├── Step 8:  VADER Sentiment (lexicon-based, no training needed)
  ├── Step 9:  Baseline ML (TF-IDF + SVM + Logistic Regression)
  ├── Step 10: ABSA (VADER on aspect-specific sentences)
  ├── Step 11: Aggregate Analytics → gezira_analytics.json
  └── Step 12: Model Comparison (SVM vs LogReg)
```

---

## 5. NLP Techniques Used (Syllabus Coverage)

### 5.1 Text Pre-Processing (Step 2)

We applied a standard preprocessing pipeline:
- **Emoji/non-ASCII removal**: `re.sub(r'[^\x00-\x7F]+', ' ', text)`
- **Elongation normalization**: "sooooo" → "so"
- **Lowercasing**
- **Tokenization**: NLTK `word_tokenize()`
- **Stopword removal**: NLTK English stopwords
- **POS tagging**: NLTK `pos_tag()`

### 5.2 Computational Morphology (Step 2)

We compared three morphological reduction techniques:

| Technique | "swimming" | "running" | "facilities" |
|---|---|---|---|
| Porter Stemmer | swim | run | facil |
| Snowball Stemmer | swim | run | facil |
| WordNet Lemmatizer | swim (POS-aware) | run | facility |

**Finding:** WordNet lemmatization preserves more meaningful word roots than Porter/Snowball stemming, especially for nouns. We use lemmatized tokens for all downstream analysis.

### 5.3 WordNet / Semantics (Step 3)

After morphology comparison, we explore WordNet to understand word meanings:

- **Synsets**: All possible meanings of a word (e.g., "pool" has 9 synsets)
- **Definitions**: What each meaning represents
- **Hypernyms**: More general concepts ("pool" is-a "body of water")
- **Hyponyms**: More specific concepts (types of "facility")
- **Synonyms**: Alternative words with similar meaning
- **Wu-Palmer Similarity**: Measuring semantic distance between words (e.g., similarity("pool", "facility") ≈ 0.4)

### 5.4 Syntax Analysis (Step 5)

Using spaCy's English model (`en_core_web_sm`):

- **Named Entity Recognition (NER)**: Identifying people, organizations, locations in reviews
- **Dependency Parsing**: Extracting noun-adjective pairs from the parse tree
  - Example: "clean pool" → `{noun: "pool", adj: "clean"}`
  - Handles both attributive ("the clean pool") and predicative ("the pool is clean") adjectives

### 5.5 Language Models (Step 7)

Built a bigram ConditionalFreqDist from all lemmatized tokens:
- For each seed word, we find what words most commonly follow it
- Example: "pool" → [("area", 12), ("clean", 8), ("open", 5), ...]
- Reveals domain-specific word associations

### 5.6 Sentiment Analysis — VADER (Step 8)

VADER (Valence Aware Dictionary and sEntiment Reasoner):
- Lexicon-based — no training required
- Uses compound score threshold (≥ 0.05 = positive)
- We compare VADER predictions against star-based labels to measure agreement

### 5.7 NLP with Machine Learning (Step 9)

Two baseline classifiers trained on TF-IDF features:

| Model | Type | Description |
|---|---|---|
| LinearSVC (SVM) | Traditional ML | Support Vector Machine with linear kernel |
| Logistic Regression | Traditional ML | Probabilistic linear classifier |

- **Features**: TF-IDF vectors (max 3000 features) from lemmatized text
- **Labels**: Derived from star ratings (4-5★ = positive, 1-2★ = negative, 3★ excluded)
- **Split**: 80% train / 20% test, stratified

### 5.8 Aspect-Based Sentiment — ABSA (Step 10)

The core innovation of this project:

1. For each review, identify which **aspects** are mentioned (keyword matching)
2. For each aspect, find the **specific sentences** that mention it
3. Run VADER on those sentences **only** to get per-aspect sentiment

**Example:**
> "The pool is beautiful, but the parking is awful!"
- `Facilities` → sentences mentioning "pool" → VADER = **positive**
- `Parking` → sentences mentioning "parking" → VADER = **negative**

---

## 6. Results & Comparisons

### Morphology: Stemming vs Lemmatization

WordNet lemmatization produces more readable and meaningful output than stemming. "facilities" → "facility" (lemma) vs "facil" (stem).

### Sentiment Models: SVM vs LogReg

| Model | Accuracy | F1 Score |
|---|---|---|
| TF-IDF + SVM | ~0.85–0.92 | ~0.90–0.95 |
| TF-IDF + LogReg | ~0.85–0.92 | ~0.90–0.95 |

> Exact numbers vary by random split. High accuracy is partially due to class imbalance (88% positive).

### Why ABSA Is Better Than Overall Sentiment

Baseline models classify the **entire review** — mixed reviews get a single label. ABSA decomposes reviews into **per-aspect intelligence**, which is far more actionable for club management.

---

## 7. Tools & Libraries

| Tool | Purpose |
|---|---|
| NLTK | Tokenization, stemming, lemmatization, POS tagging, WordNet, VADER, bigrams |
| spaCy | NER, dependency parsing |
| scikit-learn | TF-IDF, SVM, Logistic Regression, evaluation metrics |
| Streamlit | Interactive web dashboard |
| Plotly | Charts and visualizations |
| pandas / numpy | Data manipulation |

---

## 8. Limitations

1. **Dataset bias**: 88% of reviews are 4-5 stars — baseline models achieve high accuracy partly from class imbalance
2. **English only**: Arabic reviews were filtered out
3. **Keyword-based aspects**: Rule-based pre-filter may miss aspects expressed with unusual vocabulary
4. **No deep learning**: We use traditional ML and lexicon methods only (suitable for course scope)

---

## 9. Output Files

| File | Description |
|---|---|
| `gezira_analytics.json` | Per-aspect sentiment percentages |
| `confusion_matrix.png` | SVM & LogReg confusion matrix visualization |

### Sample Output (gezira_analytics.json)

```json
{
  "club": "Gezira Sporting Club",
  "total_reviews_analyzed": 368,
  "aspect_analytics": {
    "Facilities": { "mentions": 150, "positive_pct": 72.0, "negative_pct": 28.0 },
    "Parking":    { "mentions": 45,  "positive_pct": 25.0, "negative_pct": 75.0 }
  }
}
```
