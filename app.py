import os
import tempfile

import streamlit as st
import whisper

from groq import Groq
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
WHISPER_MODEL  = "base"
GROQ_MODEL     = "llama3-8b-8192"
SUPPORTED_EXTS = ["wav", "mp3", "m4a"]

SUMMARY_PROMPT = """You are an expert note-taking assistant.

Create concise, structured notes from the transcript below.

Transcript:
{transcript}

Respond ONLY in this exact format (keep section headings):

## Main Topics
- <topic>

## Key Points
- <point>

## Action Items
- <action item> (write "None identified" if there are none)

## Summary
<2-3 sentence paragraph>"""


st.set_page_config(
    page_title="AI Notes Summarizer",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ AI Notes Summarizer")
st.caption("Upload a recording → get a transcript + structured notes in seconds.")


@st.cache_resource(show_spinner="Loading speech model…")
def load_whisper():
    return whisper.load_model(WHISPER_MODEL)


def get_groq_client() -> Groq | None:
    if not GROQ_API_KEY:
        st.error(
            "⚠️ **GROQ_API_KEY not found.** "
            "Add it to your `.env` file and restart the app.",
            icon="🔑",
        )
        return None
    return Groq(api_key=GROQ_API_KEY)


def transcribe(audio_path: str) -> str:
    model = load_whisper()
    result = model.transcribe(audio_path)
    return result["text"].strip()


def summarize(transcript: str, client: Groq) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript)}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


for key in ("transcript", "summary", "processed_name"):
    st.session_state.setdefault(key, None)


with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. 🎙️ Upload a **WAV / MP3 / M4A** file  
        2. 🔍 Whisper transcribes the speech  
        3. 🤖 Groq (LLaMA 3) structures the notes  
        4. 📋 Copy or download your summary  
        """
    )
    st.divider()
    st.markdown(
        f"**Models**  \n"
        f"- Speech: `openai/whisper-{WHISPER_MODEL}`  \n"
        f"- Summary: `{GROQ_MODEL}`"
    )


uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=SUPPORTED_EXTS,
    help="Supports WAV, MP3, and M4A up to ~25 MB.",
)

if uploaded_file:
    file_size = len(uploaded_file.getvalue())
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("File", uploaded_file.name)
    col_info2.metric("Size", human_size(file_size))
    col_info3.metric("Format", uploaded_file.type or uploaded_file.name.split(".")[-1].upper())

    st.audio(uploaded_file)

    already_done = st.session_state.processed_name == uploaded_file.name
    btn_label = "✅ Re-generate Notes" if already_done else "⚡ Generate Notes"
    run = st.button(btn_label, type="primary", use_container_width=True)

    if run:
        client = get_groq_client()
        if client:
            tmp_path = save_upload_to_temp(uploaded_file)
            try:
                with st.status("Transcribing audio…", expanded=True) as status:
                    st.write("Running Whisper on your file…")
                    transcript = transcribe(tmp_path)
                    word_count = len(transcript.split())
                    st.write(f"✓ Transcribed **{word_count:,} words**")

                    status.update(label="Generating notes…")
                    st.write("Sending transcript to Groq…")
                    summary = summarize(transcript, client)
                    st.write("✓ Notes ready")

                    status.update(label="Done!", state="complete", expanded=False)

                st.session_state.transcript     = transcript
                st.session_state.summary        = summary
                st.session_state.processed_name = uploaded_file.name

            except Exception as exc:
                st.error(f"Something went wrong: {exc}", icon="❌")
            finally:
                os.unlink(tmp_path)


if st.session_state.transcript:
    st.divider()
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("📝 Transcript")
        st.text_area(
            label="transcript_area",
            label_visibility="collapsed",
            value=st.session_state.transcript,
            height=420,
        )
        st.download_button(
            "⬇️ Download transcript",
            data=st.session_state.transcript,
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with right:
        st.subheader("🗒️ Generated Notes")
        st.markdown(st.session_state.summary)
        st.divider()
        st.download_button(
            "⬇️ Download notes",
            data=st.session_state.summary,
            file_name="notes_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

elif not uploaded_file:
    st.info(
        "👆 Upload an audio file above to get started. "
        "Lectures, meetings, voice memos — anything works.",
        icon="💡",
    )