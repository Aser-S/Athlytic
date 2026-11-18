"""
dashboard.py — Athlytic NLP Dashboard (Streamlit)
Shows every pipeline step's output in an interactive GUI.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, re, os
from collections import Counter
from langdetect import detect

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk import pos_tag, bigrams, ConditionalFreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import spacy
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Athlytic NLP Dashboard", page_icon="🏅", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2d 0%, #2d2d44 100%);
        border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center; border-left: 5px solid #ff4b4b;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: white; }
    .metric-title { font-size: 0.9rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 1px; }
    .step-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 1.5rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Aspect keywords (same as pipeline.py) ────────────────────────────────────
ASPECTS = {
    "Facilities": ["pool","swimming","lido","gym","fitness","court","tennis","squash",
        "football","pitch","track","field","shower","sauna","spa","golf","equipment","stadium"],
    "Prices": ["price","fee","fees","expensive","cheap","affordable","overpriced",
        "cost","membership","worth","value","money","pricey","reasonable"],
    "Community": ["community","social","atmosphere","vibe","people","members",
        "family","friends","friendly","welcoming","classy","crowd"],
    "Greenery": ["green","garden","tree","trees","grass","park","nature",
        "plants","outdoor","landscape","lawn","shade","scenery","nile"],
    "Service": ["staff","service","employee","receptionist","security","guard",
        "trainer","coach","waiter","helpful","rude","polite","management"],
    "Crowd": ["crowded","busy","packed","quiet","empty","rush","peak",
        "weekend","morning","evening","queue","wait","full"],
    "Food & Beverages": ["food","restaurant","cafe","coffee","drink","menu",
        "meal","breakfast","lunch","dinner","snack","buffet"],
    "Parking": ["parking","car","spot","garage","valet","traffic"],
    "Kids Area": ["kids","children","child","playground","play area","toddler","slide","swing"],
    "Cleanliness": ["clean","dirty","hygiene","filthy","messy","smell",
        "bathroom","toilet","trash","maintenance","maintained"],
}

# ── Run the full pipeline (cached) ───────────────────────────────────────────
@st.cache_data(show_spinner="Running NLP pipeline...")
def run_pipeline():
    """Run the entire pipeline and return all intermediate results."""
    # NLTK downloads
    for r in ['punkt','punkt_tab','stopwords','wordnet','averaged_perceptron_tagger_eng','vader_lexicon']:
        nltk.download(r, quiet=True)

    nlp = spacy.load("en_core_web_sm")
    sia = SentimentIntensityAnalyzer()

    # 1. Load data
    with open("data/gezira_reviews.json", "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open("data/gezira_lowest500.json", "r", encoding="utf-8") as f:
        data2 = json.load(f)

    raw_data = data1 + data2
    df = pd.DataFrame(raw_data)
    
    # Clean and filter
    df = df[df["text"].notna()]
    df = df.drop_duplicates(subset=["text"])
    
    def is_english(text):
        try:
            return detect(text) == 'en'
        except:
            return False

    df = df[df["text"].apply(is_english)].reset_index(drop=True)
    df["stars"] = df["stars"].astype(int)

    # 2. Preprocessing
    stop_words = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")

    def pos2wn(tag):
        if tag.startswith("J"): return "a"
        if tag.startswith("V"): return "v"
        if tag.startswith("N"): return "n"
        if tag.startswith("R"): return "r"
        return "n"

    def preprocess(text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = text.lower().strip()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        tagged = pos_tag(tokens)
        lemmas = [lem.lemmatize(w, pos2wn(t)) for w, t in tagged]
        return {"tokens": tokens, "pos_tags": tagged, "lemmas": lemmas,
                "porter": [porter.stem(w) for w in tokens],
                "snowball": [snowball.stem(w) for w in tokens]}

    pp = df["text"].apply(preprocess)
    df["tokens"] = pp.apply(lambda x: x["tokens"])
    df["lemmas"] = pp.apply(lambda x: x["lemmas"])
    df["pos_tags"] = pp.apply(lambda x: x["pos_tags"])
    df["porter_stems"] = pp.apply(lambda x: x["porter"])
    df["snowball_stems"] = pp.apply(lambda x: x["snowball"])

    # 4. Competitors
    COMPETITORS = ["gezira","wadi degla","ahly","al ahly","al-ahly","zamalek",
        "shooting club","maadi","national club","helio lido","sporting",
        "smouha","ittihad","olympic","heliopolis","shams"]
    df["competitor_mentions"] = df["text"].apply(
        lambda t: [c for c in COMPETITORS if re.search(r'\b'+re.escape(c)+r'\b', t.lower())])

    # 5. Syntax (spaCy)
    def extract_syntax(text):
        doc = nlp(text)
        ents = [{"text": e.text, "label": e.label_} for e in doc.ents]
        pairs = []
        for tok in doc:
            if tok.pos_ == "ADJ" and tok.head.pos_ == "NOUN":
                pairs.append({"noun": tok.head.text.lower(), "adj": tok.text.lower()})
            if tok.dep_ in ("acomp","attr") and tok.pos_ == "ADJ":
                for ch in tok.head.children:
                    if ch.dep_ == "nsubj" and ch.pos_ == "NOUN":
                        pairs.append({"noun": ch.text.lower(), "adj": tok.text.lower()})
        return {"entities": ents, "noun_adj_pairs": pairs}

    syn = df["text"].apply(extract_syntax)
    df["entities"] = syn.apply(lambda x: x["entities"])
    df["noun_adj_pairs"] = syn.apply(lambda x: x["noun_adj_pairs"])

    # 6. Aspects
    def match_aspects(tokens):
        ts = set(tokens); tj = " ".join(tokens); matched = []
        for asp, kws in ASPECTS.items():
            for kw in kws:
                if (" " in kw and kw in tj) or (kw in ts):
                    matched.append(asp); break
        return matched
    df["aspects"] = df["lemmas"].apply(match_aspects)

    # 7. Bigrams
    all_lemmas = [l for ll in df["lemmas"] for l in ll]
    cfd = ConditionalFreqDist(bigrams(all_lemmas))
    bigram_data = {}
    for w in ["pool","staff","parking","food","green","kid","clean","price","time","place","court","service","member"]:
        if w in cfd: bigram_data[w] = cfd[w].most_common(10)

    # 8. VADER
    df["vader_sentiment"] = df["text"].apply(
        lambda t: "positive" if sia.polarity_scores(t)["compound"] >= 0.05 else "negative")
    df["star_label"] = df["stars"].apply(lambda s: "positive" if s >= 4 else "negative")

    # 9. ML
    ml_df = df[df["stars"] != 3].copy()
    ml_df["label"] = ml_df["stars"].apply(lambda s: 1 if s >= 4 else 0)
    X_text = ml_df["lemmas"].apply(lambda l: " ".join(l))
    y = ml_df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)
    tfidf = TfidfVectorizer(max_features=3000)
    X_tr_tf = tfidf.fit_transform(X_tr); X_te_tf = tfidf.transform(X_te)
    svm = LinearSVC(random_state=42, max_iter=5000); svm.fit(X_tr_tf, y_tr)
    lr = LogisticRegression(random_state=42, max_iter=1000); lr.fit(X_tr_tf, y_tr)
    svm_pred = svm.predict(X_te_tf); lr_pred = lr.predict(X_te_tf)

    ml_results = {
        "svm_report": classification_report(y_te, svm_pred, target_names=["negative","positive"], output_dict=True),
        "lr_report": classification_report(y_te, lr_pred, target_names=["negative","positive"], output_dict=True),
        "svm_cm": confusion_matrix(y_te, svm_pred).tolist(),
        "lr_cm": confusion_matrix(y_te, lr_pred).tolist(),
        "svm_acc": accuracy_score(y_te, svm_pred), "svm_f1": f1_score(y_te, svm_pred),
        "lr_acc": accuracy_score(y_te, lr_pred), "lr_f1": f1_score(y_te, lr_pred),
    }

    # 10. ABSA
    def absa(text, aspects_matched):
        results = {}; sents = sent_tokenize(text)
        for asp in aspects_matched:
            kws = ASPECTS.get(asp, [])
            rel = [s for s in sents if any(kw in s.lower() for kw in kws)]
            if not rel: results[asp] = "unknown"; continue
            sc = sia.polarity_scores(" ".join(rel))["compound"]
            results[asp] = "positive" if sc >= 0.05 else "negative"
        return results

    samp = df[df["aspects"].apply(len) > 0].copy()
    samp["absa_results"] = samp.apply(lambda r: absa(r["text"], r["aspects"]), axis=1)

    # 11. Aggregate
    aspect_analytics = {}
    for asp in ASPECTS:
        ar = samp[samp["absa_results"].apply(lambda d: asp in d)]
        if len(ar) == 0: continue
        ss = ar["absa_results"].apply(lambda d: d[asp])
        p = int((ss == "positive").sum()); n = int((ss == "negative").sum()); t = p + n
        aspect_analytics[asp] = {"mentions": t,
            "positive_pct": round(p/t*100,1) if t else 0,
            "negative_pct": round(n/t*100,1) if t else 0}

    return df, samp, bigram_data, ml_results, aspect_analytics


# ── Run pipeline ─────────────────────────────────────────────────────────────
try:
    df, samp, bigram_data, ml_results, aspect_analytics = run_pipeline()
    pipeline_ok = True
except Exception as e:
    st.error(f"Pipeline error: {e}")
    pipeline_ok = False

if not pipeline_ok:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏅 Athlytic — NLP Pipeline Dashboard")
st.caption("Gezira Sporting Club · Google Maps Review Analysis")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Data Overview",
    "🔤 Preprocessing",
    "📖 WordNet",
    "🏟️ Competitors",
    "🌐 Syntax (NER)",
    "🎯 Aspects",
    "📊 Bigrams",
    "🔍 ABSA Results",
    "📝 Reviews",
])

# ── Tab 0: Data Overview ─────────────────────────────────────────────────────
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", len(df))
    c2.metric("Avg Stars", f"{df['stars'].mean():.1f}")
    c3.metric("5-Star %", f"{(df['stars']==5).mean()*100:.0f}%")

    fig = px.histogram(df, x="stars", nbins=5, color_discrete_sequence=["#667eea"],
                       title="Star Rating Distribution")
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 1: Preprocessing ─────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Text Preprocessing — Tokenization, Stemming & Lemmatization")
    idx = st.slider("Pick a review index", 0, len(df)-1, 0, key="pp_idx")
    row = df.iloc[idx]
    st.markdown(f"**Original:** {row['text']}")

    n = min(10, len(row["tokens"]))
    comp_df = pd.DataFrame({
        "Token": row["tokens"][:n],
        "Porter Stem": row["porter_stems"][:n],
        "Snowball Stem": row["snowball_stems"][:n],
        "WordNet Lemma": row["lemmas"][:n],
    })
    st.table(comp_df)

    st.markdown("**POS Tags** (first 10):")
    st.write(row["pos_tags"][:10])

# ── Tab 2: WordNet ───────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📖 WordNet Explorer")
    word = st.text_input("Enter a word to explore:", value="pool")
    if word:
        synsets = wordnet.synsets(word)
        if not synsets:
            st.warning(f"No synsets found for '{word}'")
        else:
            st.success(f"**{len(synsets)}** synset(s) found for **'{word}'**")
            for ss in synsets[:4]:
                with st.expander(f"{ss.name()} [{ss.pos()}]"):
                    st.write(f"**Definition:** {ss.definition()}")
                    if ss.examples():
                        st.write(f"**Example:** _{ss.examples()[0]}_")
                    hyper = ss.hypernyms()
                    if hyper:
                        st.write(f"**Hypernym (is-a):** {hyper[0].name()} — {hyper[0].definition()}")
                    hypo = ss.hyponyms()
                    if hypo:
                        st.write(f"**Hyponyms (types):** {', '.join(h.name().split('.')[0] for h in hypo[:5])}")

            all_syns = set()
            for ss in synsets:
                for lm in ss.lemmas():
                    if lm.name().lower() != word.lower():
                        all_syns.add(lm.name().replace("_"," "))
            if all_syns:
                st.info(f"**Synonyms:** {', '.join(list(all_syns)[:8])}")

        # Similarity
        st.markdown("---")
        st.subheader("Wu-Palmer Similarity")
        c1, c2 = st.columns(2)
        w1 = c1.text_input("Word 1", "pool")
        w2 = c2.text_input("Word 2", "facility")
        s1, s2 = wordnet.synsets(w1), wordnet.synsets(w2)
        if s1 and s2:
            sim = s1[0].wup_similarity(s2[0])
            if sim: st.metric("Similarity", f"{sim:.3f}")

# ── Tab 3: Competitors ──────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🏟️ Competitor Mentions")
    comp_df = df[df["competitor_mentions"].apply(len) > 0]
    st.metric("Reviews mentioning competitors", len(comp_df))
    for _, r in comp_df.head(8).iterrows():
        with st.expander(f"⭐{r['stars']} — mentions: {', '.join(r['competitor_mentions'])}"):
            st.write(r["text"])

# ── Tab 4: Syntax ────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("🌐 Named Entities & Noun-Adj Pairs (spaCy)")
    idx2 = st.slider("Pick a review", 0, len(df)-1, 0, key="syn_idx")
    r = df.iloc[idx2]
    st.markdown(f"**Review:** {r['text']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Named Entities:**")
        if r["entities"]:
            st.table(pd.DataFrame(r["entities"]))
        else:
            st.write("None found")
    with c2:
        st.markdown("**Noun-Adjective Pairs:**")
        if r["noun_adj_pairs"]:
            st.table(pd.DataFrame(r["noun_adj_pairs"]))
        else:
            st.write("None found")

# ── Tab 5: Aspects ───────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🎯 Aspect Keyword Matching")
    all_asp = [a for al in df["aspects"] for a in al]
    ac = Counter(all_asp).most_common()
    if ac:
        adf = pd.DataFrame(ac, columns=["Aspect","Count"]).sort_values("Count", ascending=True)
        fig = px.bar(adf, y="Aspect", x="Count", orientation="h",
                     color="Count", color_continuous_scale="Viridis", template="plotly_dark",
                     title="Aspect Mention Frequency")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.metric("Reviews with no aspect", sum(1 for a in df["aspects"] if not a))

# ── Tab 6: Bigrams ──────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("📊 Bigram Word Associations")
    st.write("What word most commonly follows a given seed word?")
    for word, pairs in bigram_data.items():
        with st.expander(f"'{word}' →"):
            bdf = pd.DataFrame(pairs, columns=["Following Word","Count"])
            st.table(bdf)


# ── Tab 7: ABSA ─────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("🔍 Aspect-Based Sentiment Analysis")
    st.write("Sentiment is calculated **per aspect** using VADER on the specific sentences mentioning each aspect.")

    if aspect_analytics:
        aspects, pos_p, neg_p, ment = [], [], [], []
        for a, s in aspect_analytics.items():
            if s["mentions"] > 0:
                aspects.append(a); pos_p.append(s["positive_pct"])
                neg_p.append(s["negative_pct"]); ment.append(s["mentions"])

        pdf = pd.DataFrame({"Aspect":aspects,"Positive %":pos_p,"Negative %":neg_p,"Mentions":ment})
        pdf = pdf.sort_values("Mentions", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(y=pdf["Aspect"], x=pdf["Negative %"], name="Negative",
            orientation="h", marker_color="#ff4b4b",
            text=pdf["Negative %"].apply(lambda x: f"{x:.0f}%" if x > 5 else ""), textposition="inside"))
        fig.add_trace(go.Bar(y=pdf["Aspect"], x=pdf["Positive %"], name="Positive",
            orientation="h", marker_color="#00cc96",
            text=pdf["Positive %"].apply(lambda x: f"{x:.0f}%" if x > 5 else ""), textposition="inside"))
        fig.update_layout(barmode="stack", template="plotly_dark", height=450,
            title="Sentiment per Aspect", xaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Sample ABSA Results:**")
    for _, r in samp.head(5).iterrows():
        with st.expander(f"⭐{r['stars']} — {r['text'][:60]}..."):
            for asp, sent in r["absa_results"].items():
                e = "✅" if sent == "positive" else "❌"
                st.write(f"{e} **{asp}**: {sent}")

# ── Tab 8: Review Explorer ─────────────────────────────────────────────────
with tabs[8]:
    st.subheader("📝 Raw Review Explorer")
    c1, c2 = st.columns(2)
    with c1:
        star_range = st.slider("Filter by stars", 1, 5, (1,5))
    with c2:
        q = st.text_input("Search keywords", "")

    filt = df[(df["stars"] >= star_range[0]) & (df["stars"] <= star_range[1])]
    if q:
        filt = filt[filt["text"].str.contains(q, case=False, na=False)]

    st.write(f"Showing {len(filt)} reviews:")
    for _, r in filt.head(15).iterrows():
        with st.expander(f"{'⭐'*int(r['stars'])} — {r.get('publishAt','')[:10]}"):
            st.write(f"**{r.get('name','Anonymous')}**")
            st.write(r["text"])

st.markdown("---")
st.caption("Athlytic — NLP Pipeline Dashboard · CSE389 Natural Language Processing")
