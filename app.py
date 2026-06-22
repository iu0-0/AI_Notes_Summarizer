import os
import tempfile
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WHISPER_MODEL = "whisper-large-v3"
GROQ_MODEL = "llama-3.1-8b-instant"
SUPPORTED_EXTS = ["wav", "mp3", "m4a", "webm"]

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
- <action item>

## Summary
<2-3 sentence paragraph>
"""

st.set_page_config(
    page_title="AI Notes Summarizer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

.stApp{
    background:#FDF6FF;
}

html, body, [class*="css"]{
    font-family:'Press Start 2P', cursive;
}

.hero{
    background:#CDB4FF;
    border:4px solid black;
    padding:30px;
    text-align:center;
    box-shadow:8px 8px 0px black;
    margin-bottom:25px;
}

.hero h1{
    color:#3D2C5A;
    font-size:30px;
}

.hero p{
    color:#4B3F72;
    font-size:11px;
}

section[data-testid="stSidebar"]{
    background:#FFE5EC;
}

.stButton > button{
    width:100%;
    height:60px;
    background:#FFAFCC;
    color:black;
    border:3px solid black;
    border-radius:0px;
    box-shadow:5px 5px 0px black;
    font-weight:bold;
}

.stButton > button:hover{
    transform:translate(2px,2px);
    box-shadow:3px 3px 0px black;
}

.stDownloadButton > button{
    width:100%;
    background:#BDE0FE;
    color:black;
    border:3px solid black;
    border-radius:0px;
    box-shadow:4px 4px 0px black;
}

[data-testid="stMetric"]{
    background:white;
    border:3px solid black;
    padding:12px;
    box-shadow:4px 4px 0px black;
}

textarea{
    border:3px solid black !important;
}

.stTextArea textarea{
    background:white !important;
    color:black !important;
}

.footer{
    text-align:center;
    color:#666;
    padding:20px;
    margin-top:30px;
    font-size:10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🎙 AI Notes Summarizer</h1>
<p>Convert meetings, lectures, and voice recordings into structured notes instantly</p>
</div>
""", unsafe_allow_html=True)

for key in ("transcript", "summary", "processed_name"):
    st.session_state.setdefault(key, None)


def get_groq_client():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY missing.")
        return None
    return Groq(api_key=GROQ_API_KEY)


def save_upload_to_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def transcribe(audio_path, client):
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="text"
        )

    return response.strip() if isinstance(response, str) else response.text.strip()


def summarize(transcript, client):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": SUMMARY_PROMPT.format(
                    transcript=transcript
                )
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def human_size(size):
    for unit in ["B","KB","MB","GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


with st.sidebar:

    st.title("⚙ Settings")

    st.markdown("---")

    st.markdown("### AI Models")

    st.success(f"Speech: {WHISPER_MODEL}")

    st.success(f"Summary: {GROQ_MODEL}")

    st.markdown("---")

    st.markdown(
        """
### Workflow

1️⃣ Upload Audio

2️⃣ Speech → Text

3️⃣ AI Summarization

4️⃣ Download Notes
"""
    )

    st.markdown("---")

    st.info(
        "Supports lectures, meetings, interviews, podcasts, and voice memos."
    )

st.markdown("### 📤 Upload Audio")

uploaded_file = st.file_uploader(
    "",
    type=SUPPORTED_EXTS
)

if uploaded_file:

    file_size = len(uploaded_file.getvalue())

    c1, c2, c3 = st.columns(3)

    c1.metric("File Name", uploaded_file.name)

    c2.metric("File Size", human_size(file_size))

    c3.metric("Format", uploaded_file.name.split(".")[-1].upper())

    st.audio(uploaded_file)

    if st.button(
        "⚡ Generate Notes",
        use_container_width=True
    ):

        client = get_groq_client()

        if client:

            temp_path = save_upload_to_temp(
                uploaded_file
            )

            try:

                with st.spinner(
                    "🎧 Transcribing audio..."
                ):

                    transcript = transcribe(
                        temp_path,
                        client
                    )

                with st.spinner(
                    "🤖 Generating notes..."
                ):

                    summary = summarize(
                        transcript,
                        client
                    )

                st.session_state.transcript = transcript
                st.session_state.summary = summary
                st.session_state.processed_name = uploaded_file.name

                st.success(
                    "Notes generated successfully!"
                )

            except Exception as e:

                st.error(str(e))

            finally:

                os.unlink(temp_path)

if st.session_state.transcript:

    st.markdown("---")

    transcript_words = len(
        st.session_state.transcript.split()
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Words",
        f"{transcript_words:,}"
    )

    m2.metric(
        "Characters",
        f"{len(st.session_state.transcript):,}"
    )

    m3.metric(
        "Status",
        "Completed"
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.markdown(
            "## 📜 Transcript"
        )

        st.text_area(
            "",
            st.session_state.transcript,
            height=500
        )

        st.download_button(
            "⬇ Download Transcript",
            st.session_state.transcript,
            "transcript.txt",
            "text/plain",
            use_container_width=True
        )

    with right:

        st.markdown(
            "## 📝 Generated Notes"
        )

        st.markdown(
            st.session_state.summary
        )

        st.download_button(
            "⬇ Download Notes",
            st.session_state.summary,
            "notes_summary.txt",
            "text/plain",
            use_container_width=True
        )

else:

    st.info(
        "Upload an audio file to get started."
    )

st.markdown(
    """
<div class="footer">
Built with ❤️ using Streamlit + Groq + Whisper
</div>
""",
unsafe_allow_html=True
)

