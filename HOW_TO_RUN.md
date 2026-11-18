# How to Run — Athlytic NLP Pipeline

## Prerequisites

- **Anaconda** (or Miniconda) installed
- Python 3.10+

---

## Step-by-Step Setup

### 1. Open Anaconda Prompt and activate the base environment

```bash
conda activate base
```

> If you're using a custom environment, replace `base` with your environment name.  
> To verify: `python --version` should show 3.10+.

### 2. Navigate to the project folder

```bash
cd Athlytic
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 5. Download NLTK data (automatic)

The pipeline downloads NLTK data automatically on first run. If you want to do it manually:

```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')
```

---

## Running the Project

### Option A: Run the pipeline script (generates output files)

```bash
python pipeline.py
```

This processes all reviews and saves:
- `confusion_matrix.png` — SVM & LogReg confusion matrices
- `gezira_analytics.json` — per-aspect sentiment analytics

### Option B: Launch the interactive dashboard (recommended)

```bash
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

> The dashboard runs the full pipeline internally (cached), so you don't need to run `pipeline.py` first.

---

## What Each File Does

| File | Purpose |
|---|---|
| `data/` | Dataset — Google Maps reviews (anonymised: text, stars, date only) |
| `pipeline.py` | The full NLP pipeline — runs all 12 steps in the terminal |
| `dashboard.py` | Streamlit GUI — shows every pipeline step interactively |
| `requirements.txt` | Python package dependencies |
| `REPORT.md` | Academic report documenting the project |
| `HOW_TO_RUN.md` | This file |
| `confusion_matrix.png` | Generated — ML model confusion matrices |
| `gezira_analytics.json` | Generated — per-aspect sentiment JSON |

---

## Pipeline Steps (what runs)

| # | Step | NLP Topic Covered |
|---|---|---|
| 1 | Load Data | — |
| 2 | Text Preprocessing | Tokenization, Stopwords, POS Tagging |
| 3 | WordNet Explorer | Semantics — synsets, hypernyms, similarity |
| 4 | Competitor Detection | Regex pattern matching |
| 5 | Syntax Analysis | spaCy NER, Dependency Parsing |
| 6 | Aspect Matching | Keyword-based semantic matching |
| 7 | Bigram Model | Language Models (ConditionalFreqDist) |
| 8 | VADER Sentiment | Lexicon-based sentiment |
| 9 | Baseline ML | TF-IDF + SVM + Logistic Regression |
| 10 | ABSA | Aspect-Based Sentiment (VADER per sentence) |
| 11 | Aggregate Analytics | Save results to JSON |
| 12 | Model Comparison | SVM vs LogReg accuracy & F1 |

---

## Troubleshooting

- **`ModuleNotFoundError`**: Run `pip install -r requirements.txt` again
- **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
- **NLTK data missing**: The script downloads it automatically; check your internet connection
- **Streamlit won't open**: Try `python -m streamlit run dashboard.py`
