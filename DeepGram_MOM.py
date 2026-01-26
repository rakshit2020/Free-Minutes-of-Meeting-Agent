import os
from pathlib import Path
from deepgram import DeepgramClient
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class MeetingMinutesGenerator:
    def __init__(self):
        self.deepgram = DeepgramClient(api_key=st.secrets["DEEPGRAM_API_KEY"])
        self.nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=st.secrets["NVIDIA_API_KEY"]
        )
    
    def transcribe_audio(self, audio_file_path, model="nova-2"):
        """Transcribe audio using Deepgram"""
        print(f"\n🎤 Transcribing with Deepgram ({model})...")
        print(f"📁 File: {audio_file_path}")
        
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.2f} MB")
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                print("⏳ Processing...")
                
                # Minimal working parameters
                response = self.deepgram.listen.v1.media.transcribe_file(
                    request=audio_file.read(),
                    model=model,
                    smart_format=True,
                    punctuate=True,
                    diarize=True,
                )
            
            # Extract data
            transcript = response.results.channels[0].alternatives[0].transcript
            
            metadata = {
                "duration": response.metadata.duration,
                "confidence": response.results.channels[0].alternatives[0].confidence,
                "words_count": len(transcript.split())
            }
            
            print(f"\n✅ Transcription complete!")
            print(f"⏱️  Duration: {metadata['duration']:.2f} seconds")
            print(f"📊 Confidence: {metadata['confidence']:.2%}")
            print(f"📝 Words: {metadata['words_count']}")
            
            return {
                "transcript": transcript,
                "metadata": metadata,
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def generate_meeting_minutes(self, transcription_data):
        """Generate meeting minutes using NVIDIA LLM"""
        print("\n📝 Generating meeting minutes...")
        
        transcript = transcription_data["transcript"]
        
        prompt = f"""You are an expert executive assistant.
        Analyze the meeting transcript below and provide a **comprehensive yet concise** Minutes of Meeting (MoM).
        Focus on actionable insights and clear accountability.

        TRANSCRIPT:
        {transcript}

        Structure the output as follows:

        1. **Meeting Metadata**
        - Date & Time (if available)
        - Participants (list all speakers identified)
        - Duration

        2. **Meeting Objective**: One or two clear sentences stating the purpose.

        3. **Key Discussion Points**: 
        - Bullet points of critical topics discussed
        - Include speaker attribution for important statements (e.g., "John raised concerns about...")

        4. **Decisions Made**:
        - Clear, numbered list of concrete decisions reached
        - Distinguish from general discussions

        5. **Action Items**:
        - Format: [Task] - [Owner/Assignee] - [Deadline/Timeframe]
        - If owner or deadline not mentioned, note as "TBD"
        - Prioritize by urgency if possible

        6. **Open Questions/Blockers**:
        - Items requiring follow-up or resolution
        - Uncertainties that need clarification

        7. **Next Steps/Follow-up**:
        - When is the next meeting/check-in?
        - What needs to happen before then?

        Requirements:
        - Be concise but preserve ALL business-critical and technical details
        - Use professional, direct language
        - Ensure action items are specific and measurable
        - Flag any urgent items or risks mentioned
        - Use markdown formatting with clear hierarchy
        - If transcript quality is poor or context is missing, note uncertainties"""


        completion = self.nvidia_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            top_p=0.95,
            max_tokens=8192,
            seed=42,
            stream=True
        )
        
        meeting_minutes = []
        print("\n" + "="*80)
        print("MEETING MINUTES")
        print("="*80 + "\n")
        
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                meeting_minutes.append(content)
                print(content, end="", flush=True)
        
        print("\n" + "="*80)
        return "".join(meeting_minutes)
    
    def save_outputs(self, transcript_data, minutes, output_dir="meeting_outputs"):
        """Save files"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save transcript
        transcript_file = f"{output_dir}/transcript_{timestamp}.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write("MEETING TRANSCRIPT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Duration: {transcript_data['metadata']['duration']:.2f}s\n")
            f.write(f"Confidence: {transcript_data['metadata']['confidence']:.2%}\n\n")
            f.write("="*80 + "\n\n")
            f.write(transcript_data["transcript"])
        
        print(f"\n✅ Transcript: {transcript_file}")
        
        # Save minutes
        minutes_file = f"{output_dir}/minutes_{timestamp}.md"
        with open(minutes_file, "w", encoding="utf-8") as f:
            f.write(f"# Meeting Minutes - {timestamp}\n\n")
            f.write(minutes)
        
        print(f"✅ Minutes: {minutes_file}")


def main():
    print("\n" + "="*80)
    print("🎯 MEETING MINUTES GENERATOR")
    print("="*80)
    
    generator = MeetingMinutesGenerator()
    
    audio_file = input("\n📁 Audio file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(audio_file):
        print(f"❌ Not found: {audio_file}")
        return
    
    print("\n🤖 Models: 1=nova-2  2=nova-3  3=whisper")
    model_choice = input("Select (default=1): ").strip() or "1"
    models = {"1": "nova-2", "2": "nova-3", "3": "whisper-large"}
    model = models.get(model_choice, "nova-2")
    
    try:
        # Transcribe
        transcript_data = generator.transcribe_audio(audio_file, model=model)
        
        # Preview
        print("\n📝 Preview:")
        print("-" * 80)
        print(transcript_data["transcript"][:500] + "...")
        print("-" * 80)
        
        # Generate minutes
        minutes = generator.generate_meeting_minutes(transcript_data)
        
        # Save
        generator.save_outputs(transcript_data, minutes)
        
        print("\n✅ DONE! Check 'meeting_outputs' folder.\n")
        
    except Exception as e:
        print(f"\n❌ {e}")


if __name__ == "__main__":
    main()
