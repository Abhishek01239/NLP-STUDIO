import streamlit as st

st.set_page_config(
    page_title="NLP Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #333;
        margin-top: 1rem;
    }
    .label-positive { color: #4ade80; font-weight: 700; font-size: 1.3rem; }
    .label-negative { color: #f87171; font-weight: 700; font-size: 1.3rem; }
    .label-neutral  { color: #facc15; font-weight: 700; font-size: 1.3rem; }
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .chat-user { background:#2d2d3e; border-radius:10px; padding:10px 14px; margin:6px 0; text-align:right; }
    .chat-bot  { background:#1a1a2e; border-radius:10px; padding:10px 14px; margin:6px 0; border-left:3px solid #667eea; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 NLP Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sentiment Analysis · Text Summarization · AI Chatbot</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Sentiment Analysis", "📝 Text Summarization", "🤖 AI Chatbot"])

# ── TAB 1: SENTIMENT ──────────────────────────────────────────────────────────
with tab1:
    from sentiment import analyze_sentiment, get_word_sentiments
    import pandas as pd

    st.subheader("💬 Sentiment Analysis")
    st.markdown("Detect the emotional tone of any text using a fine-tuned transformer model.")

    sample_texts = [
        "Select a sample or type your own...",
        "I absolutely loved the movie! The acting was superb and the plot was gripping.",
        "This product is terrible. Complete waste of money and terrible customer service.",
        "The package arrived on time. It does what it says.",
        "I'm so frustrated with this app. It keeps crashing every single time!",
    ]

    sample = st.selectbox("🎯 Try a sample:", sample_texts)
    user_text = st.text_area(
        "Or type your text here:",
        value="" if sample == sample_texts[0] else sample,
        height=150,
        placeholder="Enter any text to analyze its sentiment..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("Analyze →", key="sent_btn")

    if analyze_btn:
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing sentiment..."):
                result = analyze_sentiment(user_text)

            label = result["label"]
            score = result["score"]
            css_class = f"label-{label.lower()}"
            emoji = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(label, "🤔")

            st.markdown(f"""
            <div class="result-box">
                <span class="{css_class}">{emoji} {label}</span>
                <span style="color:#aaa; margin-left:1rem;">Confidence: {score:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Confidence Scores:**")
            scores_df = pd.DataFrame(result["all_scores"]).T
            scores_df.columns = ["Score"]
            st.bar_chart(scores_df)

            st.markdown("**Word-level Sentiment Hints:**")
            word_data = get_word_sentiments(user_text)
            if word_data:
                st.dataframe(pd.DataFrame(word_data), use_container_width=True, hide_index=True)
            else:
                st.info("No strong sentiment words detected.")

# ── TAB 2: SUMMARIZATION ──────────────────────────────────────────────────────
with tab2:
    from summarizer import summarize_text

    st.subheader("📝 Text Summarization")
    st.markdown("Condense long text into a concise summary using a seq2seq model.")

    sample_articles = {
        "Select a sample...": "",
        "Tech / AI": """Artificial intelligence has rapidly transformed industries across the globe.
        From healthcare to finance, AI systems are now handling tasks that once required human expertise.
        Deep learning models trained on massive datasets can now diagnose diseases from medical images with
        accuracy rivaling experienced doctors. In the financial sector, AI algorithms analyze market trends
        and execute trades in milliseconds. Natural language processing has enabled virtual assistants to
        understand and respond to human speech with remarkable fluency. However, this rapid advancement
        also raises important ethical questions about job displacement, data privacy, and the potential
        misuse of powerful AI systems. Policymakers and tech leaders are now working together to establish
        frameworks that ensure AI development remains beneficial and aligned with human values.""",
        "Climate": """Climate change is one of the most pressing challenges facing humanity today.
        Rising global temperatures are causing glaciers to melt at unprecedented rates, leading to rising
        sea levels that threaten coastal communities worldwide. Extreme weather events, including hurricanes,
        floods, and droughts, are becoming more frequent and intense. Scientists warn that without significant
        reductions in greenhouse gas emissions, the consequences could be catastrophic. Renewable energy
        sources like solar and wind power have become increasingly affordable and are being deployed at
        record rates. Electric vehicles are gaining market share, and many countries have committed to
        achieving net-zero emissions by mid-century.""",
    }

    chosen = st.selectbox("📰 Try a sample article:", list(sample_articles.keys()))
    article_text = st.text_area(
        "Paste your text here (min ~50 words):",
        value=sample_articles[chosen],
        height=250,
        placeholder="Paste a long article or paragraph..."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        min_len = st.slider("Min summary length (tokens)", 20, 100, 40)
    with col_b:
        max_len = st.slider("Max summary length (tokens)", 60, 300, 130)

    sum_btn = st.button("Summarize →", key="sum_btn")

    if sum_btn:
        if len(article_text.split()) < 30:
            st.warning("Please enter at least 30 words.")
        else:
            with st.spinner("Generating summary..."):
                summary = summarize_text(article_text, min_len, max_len)

            orig = len(article_text.split())
            summ = len(summary.split())
            c1, c2, c3 = st.columns(3)
            c1.metric("Original Words", orig)
            c2.metric("Summary Words", summ)
            c3.metric("Compression", f"{(1 - summ/orig)*100:.0f}%")

            st.markdown("**📄 Summary:**")
            st.markdown(f'<div class="result-box" style="color:#e0e0e0;line-height:1.7">{summary}</div>',
                        unsafe_allow_html=True)

# ── TAB 3: CHATBOT ────────────────────────────────────────────────────────────
with tab3:
    from chatbot import get_response

    st.subheader("🤖 AI Chatbot")
    st.markdown("Chat with an AI assistant powered by DialoGPT.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.model_history = []

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg}</div>', unsafe_allow_html=True)

    col_inp, col_send, col_clear = st.columns([6, 1, 1])
    with col_inp:
        user_msg = st.text_input("Message:", placeholder="Type here...", label_visibility="collapsed")
    with col_send:
        send_btn = st.button("Send", key="chat_send")
    with col_clear:
        clear_btn = st.button("Clear", key="chat_clear")

    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.model_history = []
        st.rerun()

    if send_btn and user_msg.strip():
        with st.spinner("Thinking..."):
            bot_reply, st.session_state.model_history = get_response(
                user_msg, st.session_state.model_history
            )
        st.session_state.chat_history.append(("user", user_msg))
        st.session_state.chat_history.append(("bot", bot_reply))
        st.rerun()