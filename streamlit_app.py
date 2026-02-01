import streamlit as st
import os
import tempfile
from fpdf import FPDF
from DeepGram_MOM import MeetingMinutesGenerator

def create_pdf(text):
    # Sanitize text for standard PDF fonts (Helvetica doesn't support curly quotes, etc.)
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "—": "-",
        "–": "-",
        "…": "...",
        "\u2022": "*", # Bullet points
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
        
    # Also handle other common non-latin1 characters by encoding/decoding
    text = text.encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    
    # Simple multi-cell for the content
    pdf.multi_cell(0, 10, text=text)
    
    # Convert bytearray to bytes for Streamlit
    return bytes(pdf.output())


st.set_page_config(
    page_title="Meeting Minutes Generator",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("📝 Meeting Minutes Generator")
    st.markdown("Upload your meeting recording and get professional minutes in seconds.")

    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox(
            "Select Model",
            options=["nova-3", "whisper-large"],
            index=0,
            help="Choose the transcription model."
        )
        st.info("Supported formats: WAV, MP3, M4A, MP4, etc.")

    # File uploader
    uploaded_file = st.file_uploader("Upload Audio/Video File", type=['wav', 'mp3', 'm4a', 'mp4', 'ogg', 'flac', 'webm'])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.audio(uploaded_file)

        if st.button("Generate Minutes", type="primary"):
            try:
                # Initialize generator
                generator = MeetingMinutesGenerator()
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Transcribe
                status_text.text("🎤 Transcribing audio... (This might take a moment)")
                progress_bar.progress(30)
                
                transcript_data = generator.transcribe_audio(tmp_file_path, model=model_choice)
                
                progress_bar.progress(60)
                status_text.text("📝 Generating meeting minutes using NVIDIA AI...")
                
                # Generate Minutes
                minutes = generator.generate_meeting_minutes(transcript_data)
                
                progress_bar.progress(100)
                status_text.success("✅ Processing Complete!")
                
                # Display Results
                st.subheader("📝Meeting Minutes")
                st.markdown(minutes)
                
                # Downloads in a row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="Download Minutes (MD)",
                        data=minutes,
                        file_name=f"minutes_{uploaded_file.name}.md",
                        mime="text/markdown"
                    )
                with col2:
                    pdf_data = create_pdf(minutes)
                    st.download_button(
                        label="Download Minutes (PDF)",
                        data=pdf_data,
                        file_name=f"minutes_{uploaded_file.name}.pdf",
                        mime="application/pdf"
                    )
                with col3:
                    st.download_button(
                        label="Download Full Transcript (TXT)",
                        data=transcript_data["transcript"],
                        file_name=f"transcript_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )

                # Cleanup
                os.unlink(tmp_file_path)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
