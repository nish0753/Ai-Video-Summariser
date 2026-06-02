import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()  # Load .env (MISTRAL_API_KEY, WHISPER_MODEL, etc.)

from utils.audio import convert_to_wav, chunk_audio, process_input
from core.transcriber import transcribe_chunk, load_model
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_conclusions,
    extract_questions,
)
from core.llm_providers import translate_to_english
from core.rag_engine import build_rag_chain, ask_question

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Summariser",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #050d1a !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #080f1e !important;
    border-right: 1px solid rgba(99, 179, 237, 0.12) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a1224; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2563eb; }

/* ── Hero Header ── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #0d1b3e 0%, #0f2957 40%, #0e3b6e 70%, #0a2540 100%);
    border: 1px solid rgba(99, 179, 237, 0.18);
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 28px;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(59,130,246,0.22) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 320px; height: 180px;
    background: radial-gradient(ellipse, rgba(14,165,169,0.15) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    color: #93c5fd;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
}
.hero h1 {
    margin: 0 0 10px 0;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e0f2fe 0%, #93c5fd 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.hero p {
    margin: 0;
    color: #7fb3d3;
    font-size: 0.95rem;
    font-weight: 400;
    max-width: 560px;
    line-height: 1.6;
}

/* ── Section label ── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 8px;
    margin-top: 24px;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(15, 28, 60, 0.6);
    border: 1px solid rgba(99, 179, 237, 0.14);
    border-radius: 14px;
    padding: 20px 22px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 16px;
}
.glass-card-accent {
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 14px;
    padding: 20px 22px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}

/* ── Result title ── */
.result-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e2f0ff;
    margin: 0 0 4px 0;
    line-height: 1.3;
}
.result-sub {
    font-size: 0.85rem;
    color: #4a7fa5;
    margin-bottom: 0;
}

/* ── Metric card ── */
.metric-card {
    background: rgba(15, 28, 60, 0.55);
    border: 1px solid rgba(99, 179, 237, 0.12);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #60a5fa;
    line-height: 1;
}
.metric-label {
    font-size: 11px;
    color: #4a7fa5;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-top: 4px;
    font-weight: 500;
}

/* ── Info chips ── */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    color: #93c5fd;
    font-size: 12px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ── Status indicator ── */
.status-dot-idle { color: #64748b; }
.status-dot-working { color: #f59e0b; }
.status-dot-done { color: #10b981; }

/* ── Sidebar ── */
.sidebar-section-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 10px;
    margin-top: 18px;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(99, 179, 237, 0.1);
    margin: 18px 0;
}

/* ── Streamlit overrides ── */
[data-testid="stRadio"] > label,
[data-testid="stSelectbox"] > label,
[data-testid="stSlider"] > label,
[data-testid="stCheckbox"] > label,
[data-testid="stTextInput"] > label,
[data-testid="stFileUploader"] > label {
    color: #7fb3d3 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(9,18,40,0.8) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 10px !important;
    color: #dbeafe !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(59,130,246,0.55) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(9,18,40,0.8) !important;
    border-color: rgba(59,130,246,0.25) !important;
    border-radius: 10px !important;
    color: #dbeafe !important;
}
[data-testid="stFileUploader"] {
    background: rgba(9,18,40,0.6) !important;
    border: 2px dashed rgba(59,130,246,0.28) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(59,130,246,0.5) !important;
    background: rgba(9,18,40,0.8) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(15, 28, 60, 0.6) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    color: #93c5fd !important;
    box-shadow: none !important;
    font-size: 13px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(37,99,235,0.15) !important;
    border-color: rgba(59,130,246,0.5) !important;
    transform: none !important;
}

/* ── Success / info / warning banners ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    font-size: 13px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] > div > div > div > div {
    color: #4a7fa5 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #2563eb !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background-color: rgba(15,28,60,0.7) !important;
    border-radius: 6px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1d4ed8 0%, #0ea5e9 100%) !important;
    border-radius: 6px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Text colors ── */
h1, h2, h3, h4, h5, h6 { color: #e2f0ff !important; }
p, li, span, div { color: #9ab5cc; }

/* ── Log area ── */
.log-container {
    background: rgba(5,10,25,0.7);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 10px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: #4ade80;
    max-height: 240px;
    overflow-y: auto;
    line-height: 1.7;
}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
for k, v in {
    "chunks": None,
    "transcript": "",
    "processing": False,
    "progress": 0,
    "status": "Idle",
    "log": [],
    "results": None,
    "url": "",
    "input_mode": "Upload audio",
    "chat_question": "",
    "chat_history": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def append_log(message: str):
    from datetime import datetime
    ts = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state["log"].append(f"[{ts}] {message}")
    st.session_state["log"] = st.session_state["log"][-200:]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.sidebar.markdown("""
        <div style='margin-bottom:24px;display:flex;align-items:center;gap:12px;'>
            <span style='font-size:26px;'>🎬</span>
            <span style='font-size:16px;font-weight:700;color:#e2f0ff;'>AI Video Summariser</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section-title'>⚙ Model Settings</div>", unsafe_allow_html=True)
    WHISPER_MODEL = st.selectbox(
        "Whisper model",
        ["small", "medium", "large"],
        index=0,
        help="Larger models are more accurate but slower. Use 'medium' for Hindi/Hinglish.",
    )
    language = st.selectbox(
        "Language",
        ["english", "hinglish", "hindi_to_english"],
        index=0,
        format_func=lambda x: {
            "english": "English",
            "hinglish": "Hinglish (Hindi + English)",
            "hindi_to_english": "Hindi (translate to English)",
        }[x],
        help="'Hindi (translate to English)' uses Groq Whisper's built-in translation — audio in Hindi, transcript in English.",
    )
    max_chunk_minutes = st.slider("Chunk length (min)", 1, 20, value=10)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'>👁 Display</div>", unsafe_allow_html=True)
    show_transcript = st.checkbox("Show full transcript", value=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'>📡 Live Status</div>", unsafe_allow_html=True)

    status_area = st.empty()
    progress_bar = st.sidebar.progress(st.session_state["progress"] / 100.0)

    mistral_key_present = bool(os.getenv("MISTRAL_API_KEY"))
    llm_provider = "mistral" if mistral_key_present else "local"

    if mistral_key_present:
        st.markdown("""<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
        border-radius:8px;padding:8px 12px;font-size:13px;color:#a7f3d0;text-align:center;'>
        ✅ Mistral configured &amp; active</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
        border-radius:8px;padding:8px 12px;font-size:12px;color:#fcd34d;'>
        ⚡ Using local LLM fallback</div>""", unsafe_allow_html=True)

    # Live log
    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'>📋 Activity Log</div>", unsafe_allow_html=True)
    log_placeholder = st.empty()

    def refresh_log():
        log_html = "<div class='log-container'>" + "<br>".join(
            f"<span style='color:#4ade80;'>{l}</span>" for l in reversed(st.session_state["log"][-30:])
        ) + "</div>"
        log_placeholder.markdown(log_html, unsafe_allow_html=True)

    refresh_log()


def update_status(msg: str, pct: int = None):
    icon = "⚙️" if pct is not None and pct < 100 else ("✅" if pct == 100 else "💤")
    st.session_state["status"] = msg
    status_area.markdown(
        f"<div style='font-size:12px;color:#7fb3d3;'>{icon} <b style='color:#93c5fd;'>{msg}</b></div>",
        unsafe_allow_html=True,
    )
    if pct is not None:
        st.session_state["progress"] = pct
        progress_bar.progress(pct / 100.0)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>✦ AI-Powered</div>
    <h1>AI Video Summariser</h1>
    <p>Transcribe any video or audio, generate intelligent summaries, surface action items &amp; key decisions — then chat with the content using RAG.</p>
</div>
""", unsafe_allow_html=True)

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>▸ Input Source</div>", unsafe_allow_html=True)

col_radio, col_spacer = st.columns([3, 5])
with col_radio:
    st.session_state["input_mode"] = st.radio(
        "Source",
        ("Upload audio", "YouTube URL"),
        index=0 if st.session_state["input_mode"] == "Upload audio" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

input_mode = st.session_state["input_mode"]
url = st.session_state["url"]

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

if input_mode == "Upload audio":
    uploaded_file = st.file_uploader(
        "Drop your audio file here",
        type=["wav", "mp3", "m4a", "webm"],
        help="Supported formats: WAV, MP3, M4A, WEBM",
    )
    if uploaded_file is not None:
        with st.spinner("Converting & chunking audio…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            wav_path = convert_to_wav(tmp_path)
            st.session_state["chunks"] = chunk_audio(wav_path, chunk_minutes=max_chunk_minutes)
            st.session_state["status"] = f"Ready — {len(st.session_state['chunks'])} chunk(s)"
            append_log(f"Uploaded '{uploaded_file.name}' → {len(st.session_state['chunks'])} chunk(s)")
            refresh_log()
else:
    url_col, btn_col = st.columns([4, 1])
    with url_col:
        url = st.text_input(
            "YouTube URL",
            value=st.session_state["url"],
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        st.session_state["url"] = url
    with btn_col:
        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
        if st.button("⬇ Fetch", use_container_width=True) and url:
            with st.spinner("Downloading & chunking…"):
                st.session_state["chunks"] = process_input(url, chunk_minutes=max_chunk_minutes)
                st.session_state["status"] = f"Ready — {len(st.session_state['chunks'])} chunk(s)"
                append_log(f"Fetched YouTube audio → {len(st.session_state['chunks'])} chunk(s)")
                refresh_log()

st.markdown("</div>", unsafe_allow_html=True)

# Chunk ready indicator
if st.session_state.get("chunks"):
    n = len(st.session_state["chunks"])
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:12px;'>"
        f"<span style='width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block;'></span>"
        f"<span style='color:#6ee7b7;font-size:13px;font-weight:500;'>Audio ready &nbsp;·&nbsp; {n} chunk{'s' if n!=1 else ''}</span></div>",
        unsafe_allow_html=True,
    )

# ── Run pipeline button ────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>▸ Process</div>", unsafe_allow_html=True)

run_col, _ = st.columns([2, 6])
with run_col:
    run_clicked = st.button("🚀  Run Pipeline", use_container_width=True)

if run_clicked:
    chunks_to_use = st.session_state.get("chunks")

    if not chunks_to_use and url:
        append_log("Auto-downloading YouTube URL…")
        refresh_log()
        with st.spinner("Downloading and chunking audio…"):
            try:
                st.session_state["chunks"] = process_input(url, chunk_minutes=max_chunk_minutes)
                chunks_to_use = st.session_state["chunks"]
                append_log(f"Auto-download done → {len(chunks_to_use)} chunk(s)")
            except Exception as e:
                append_log(f"Error: {e}")
                st.error("Failed to download/process the YouTube URL.")

    if not chunks_to_use:
        st.warning("⚠️  No audio source found. Upload a file or enter a YouTube URL first.")
    else:
        st.session_state["processing"] = True
        update_status("Loading Whisper model…", 0)
        load_model(model_name=WHISPER_MODEL)
        append_log(f"Loaded Whisper: {WHISPER_MODEL}")
        refresh_log()

        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(chunks_to_use)
        prog_area = st.empty()
        transcript_parts = [None] * total  # pre-allocate to maintain order

        def transcribe_one(args):
            idx, chunk = args
            if language == "hindi_to_english":
                text = transcribe_chunk(chunk, translate=True, language="hi")
            elif language == "hinglish":
                text = transcribe_chunk(chunk, translate=False, language="hi")
            else:
                text = transcribe_chunk(chunk, translate=False, language=None)
            return idx, text

        MAX_WORKERS = 4  # send 4 chunks to Groq simultaneously
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(transcribe_one, (i, chunk)): i
                       for i, chunk in enumerate(chunks_to_use)}

            for future in as_completed(futures):
                idx, text = future.result()
                transcript_parts[idx] = text
                completed += 1
                pct = int((completed / total) * 80)
                update_status(f"Transcribed {completed}/{total} chunks…", pct)
                prog_area.markdown(
                    f"<div style='font-size:12px;color:#4a7fa5;margin-bottom:4px;'>"
                    f"Chunk {completed} of {total} done</div>",
                    unsafe_allow_html=True,
                )
                append_log(f"✓ Chunk {completed}/{total} transcribed")
                refresh_log()

        transcript = " ".join(transcript_parts)

        # If Hindi-to-English mode: translate the raw Hindi transcript via Mistral
        if language == "hindi_to_english":
            update_status("Translating Hindi transcript to English via Mistral...", 83)
            append_log("Translating Hindi -> English via Mistral LLM...")
            refresh_log()
            try:
                transcript = translate_to_english(transcript)
                append_log("Translation complete")
            except Exception as e:
                append_log(f"Translation failed: {e} — keeping raw Hindi transcript")

        st.session_state["transcript"] = transcript
        prog_area.empty()

        update_status("Generating summary & insights…", 85)
        append_log("Generating title & summary")
        refresh_log()

        title = generate_title(transcript)
        try:
            summary = summarize(transcript, provider=("mistral" if llm_provider == "mistral" else None))
            append_log("Summary generated")
        except Exception as e:
            append_log(f"Summary failed: {e}")
            summary = "*(Summary generation failed — check log)*"

        update_status("Extracting insights…", 90)
        action_items  = extract_action_items(transcript)
        key_decisions = extract_key_decisions(transcript)
        conclusions   = extract_conclusions(transcript)
        questions     = extract_questions(transcript)
        append_log("Extracted action items, decisions, conclusions, questions")

        update_status("Building RAG chain…", 95)
        rag_chain = build_rag_chain(transcript)
        append_log("RAG chain ready")

        update_status("Complete!", 100)
        append_log("✅ Pipeline complete")
        refresh_log()

        st.session_state["processing"] = False
        st.session_state["results"] = {
            "title": title,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": key_decisions,
            "conclusions": conclusions,
            "questions": questions,
            "rag_chain": rag_chain,
            "transcript": transcript,
            "chunks": total,
            "language": language,
            "model": WHISPER_MODEL,
        }

# ── Results ────────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    st.markdown("<hr style='border:none;border-top:1px solid rgba(99,179,237,0.1);margin:28px 0;'>", unsafe_allow_html=True)

    # Success banner
    st.markdown("""
        <div style='display:flex;align-items:center;gap:10px;background:rgba(16,185,129,0.08);
        border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:12px 18px;margin-bottom:20px;'>
            <span style='font-size:18px;'>✅</span>
            <span style='color:#6ee7b7;font-size:14px;font-weight:600;'>Pipeline complete — results ready below</span>
        </div>
    """, unsafe_allow_html=True)

    # Title row + metrics
    title_col, m1, m2, m3 = st.columns([4, 1, 1, 1])
    with title_col:
        st.markdown(f"<p class='result-title'>{results['title']}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='chip'>🔊 {results['chunks']} chunk{'s' if results['chunks']!=1 else ''}</span>"
            f"<span class='chip'>🌐 {results['language'].capitalize()}</span>"
            f"<span class='chip'>🤖 {results['model']}</span>",
            unsafe_allow_html=True,
        )
    with m1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{results['chunks']}</div><div class='metric-label'>Chunks</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.1rem;'>{results['language'][:3].upper()}</div><div class='metric-label'>Language</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.1rem;'>{results['model']}</div><div class='metric-label'>Model</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Tabbed results
    tab_summary, tab_insights, tab_transcript, tab_chat = st.tabs([
        "📝  Summary", "💡  Insights", "📄  Transcript", "💬  Chat (RAG)"
    ])

    # ── Tab 1: Summary ──
    with tab_summary:
        st.markdown("<div class='glass-card-accent'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Generated Summary</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:#c7dff5;font-size:15px;line-height:1.75;'>{results['summary']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Insights ──
    with tab_insights:
        ins_col1, ins_col2 = st.columns(2)

        with ins_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("**✅ Action Items**")
            st.markdown(
                f"<div style='color:#c7dff5;font-size:14px;line-height:1.7;'>{results['action_items']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("**🔑 Key Decisions**")
            st.markdown(
                f"<div style='color:#c7dff5;font-size:14px;line-height:1.7;'>{results['key_decisions']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with ins_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("**🎯 Conclusions**")
            st.markdown(
                f"<div style='color:#c7dff5;font-size:14px;line-height:1.7;'>{results['conclusions']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if results.get("questions"):
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("**❓ Open Questions**")
                st.markdown(
                    f"<div style='color:#c7dff5;font-size:14px;line-height:1.7;'>{results['questions']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 3: Transcript ──
    with tab_transcript:
        if show_transcript:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Full Transcript</div>", unsafe_allow_html=True)
            st.text_area(
                "",
                results["transcript"],
                height=400,
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)
            dl_col, _ = st.columns([2, 6])
            with dl_col:
                st.download_button(
                    "⬇ Download transcript (.txt)",
                    results["transcript"],
                    file_name="transcript.txt",
                    use_container_width=True,
                )
        else:
            st.info("Enable 'Show full transcript' in the sidebar to view.")

    # ── Tab 4: Chat ──
    with tab_chat:
        st.markdown("<div class='section-label'>Ask Anything About This Content</div>", unsafe_allow_html=True)

        # Chat history display
        chat_container = st.container(height=400)
        with chat_container:
            if not st.session_state["chat_history"]:
                st.info("👋 Hi! I've read the transcript. Ask me anything about the video!")
            
            for entry in st.session_state["chat_history"]:
                role = entry["role"]
                with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
                    st.write(entry["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the video..."):
            # Display user message instantly
            with chat_container:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(prompt)
            
            st.session_state["chat_history"].append({"role": "user", "content": prompt})

            # Display assistant thinking & response
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        answer = ask_question(results["rag_chain"], st.session_state["chat_history"])
                    st.write(answer)
            
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
