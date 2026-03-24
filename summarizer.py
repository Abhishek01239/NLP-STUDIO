import streamlit as st
from transformers import pipeline

def load_summarizer():
    return pipeline(
        "summarization",
        model = "sshleifer/distilbart-cnn-12-6"
    )

def summarize_text(text:str,min_length: int = 40, max_length: int = 130)-> str:
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

    words = text.split()

    if len(words) >800:
        text = " ".join(words[:800])

    max_length = min(max_length, len(words)-1)

    min_length = min(min_length, max_length-5)

    result = summarizer(
        text,
        max_length = max_length,
        min_length = min_length,
        do_sample = False
    )

    return result[0]["summary_text"]