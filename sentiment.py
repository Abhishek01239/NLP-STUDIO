import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True
    )


LABEL_MAP = {
    "positive": "POSITIVE",
    "negative": "NEGATIVE",
    "neutral":  "NEUTRAL",
}

POSITIVE_WORDS = {
    "love", "great", "excellent", "amazing", "wonderful", "fantastic", "good",
    "best", "happy", "awesome", "perfect", "beautiful", "brilliant", "superb",
    "outstanding", "enjoyed", "glad", "pleased", "satisfied", "recommended",
    "incredible", "impressive", "delightful", "thankful", "grateful", "positive",
}

NEGATIVE_WORDS = {
    "hate", "terrible", "awful", "horrible", "bad", "worst", "poor", "ugly",
    "disgusting", "disappointing", "useless", "broken", "failed", "wrong",
    "frustrated", "annoyed", "angry", "pathetic", "waste", "avoid",
    "dreadful", "painful", "offensive", "ridiculous", "unacceptable", "miserable",
}


def analyze_sentiment(text: str) -> dict:
    """
    Returns:
        {
          "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
          "score": float,
          "all_scores": { "POSITIVE": float, "NEGATIVE": float, "NEUTRAL": float }
        }
    """
    model = load_sentiment_model()
    raw = model(text[:512])[0]   # trim to 512 chars to stay within token limit

    all_scores = {}
    best_label, best_score = "", 0.0

    for item in raw:
        label = LABEL_MAP.get(item["label"].lower(), item["label"].upper())
        all_scores[label] = round(item["score"], 4)
        if item["score"] > best_score:
            best_score = item["score"]
            best_label = label

    return {
        "label":      best_label,
        "score":      best_score,
        "all_scores": all_scores,
    }


def get_word_sentiments(text: str) -> list:
    """
    Scans each word against positive/negative keyword sets.
    Returns a list of dicts: [{"Word": ..., "Hint": ..., "Category": ...}]
    """
    words = text.split()
    results = []

    for w in words:
        clean = w.strip(".,!?\"'();:-").lower()
        if clean in POSITIVE_WORDS:
            results.append({
                "Word":     w,
                "Hint":     "Positive 😊",
                "Category": "positive"
            })
        elif clean in NEGATIVE_WORDS:
            results.append({
                "Word":     w,
                "Hint":     "Negative 😞",
                "Category": "negative"
            })

    return results