# Minutes of Meeting (MoM) Automation Agent

An end-to-end **Minutes of Meeting (MoM) automation tool** that converts meeting audio into clear, structured meeting minutes using ASR and LLMs.

This project is built as a **lightweight, fully functional prototype** to demonstrate automated transcription and MoM generation in a simple, usable workflow.

---

## 🔗 Live Demo
👉 https://minutes-of-meeting-agent-znkz7pw3ppca6cac72cgt3.streamlit.app/

---

## 🚀 What It Does
- Takes a **Google Meet audio recording** as input  
- Automatically **transcribes the audio** using ASR  
- Generates **concise, structured MoM** using an LLM  
- Outputs:
  - Meeting summary  
  - Attendees  
  - Agenda / topics discussed  
  - Decisions  
  - Action items / next steps  

---

## 🧩 How It Works
1. Record Google Meet audio using a **Chrome extension** (workaround for free Meet plans)
2. Upload the audio file via the web UI
3. Select the ASR model (default: **Deepgram Nova-2**)
4. The pipeline runs:
   - Speech-to-text (ASR)
   - LLM-based MoM generation
5. Get a clean, readable MoM instantly

---

## 📌 Notes
- This is an **initial version**, but fully working end-to-end
- Designed to be simple, fast, and easy to evaluate
- Feedback and suggestions are welcome to improve future versions

---

*Built to explore practical MoM automation using ASR + LLMs.*
