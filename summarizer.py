import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )


def summarize_text(text: str, min_length: int = 40, max_length: int = 130) -> str:
    """
    Summarizes the given text.

    Args:
        text       : input article or paragraph
        min_length : minimum number of tokens in the summary
        max_length : maximum number of tokens in the summary

    Returns:
        summary string
    """
    summarizer = load_summarizer()

    # ── 1. Trim very long inputs ──────────────────────────────────────────────
    words = text.split()
    if len(words) > 800:
        text = " ".join(words[:800])

    # ── 2. Guard: max_length must be less than input length ───────────────────
    max_length = min(max_length, len(words) - 1)

    # ── 3. Guard: min_length must be less than max_length ────────────────────
    min_length = min(min_length, max_length - 5)

    # ── 4. Run summarization ──────────────────────────────────────────────────
    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

    return result[0]["summary_text"]