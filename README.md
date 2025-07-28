Title: ZeroPhish Gate 
colorFrom: purple
colorTo: red
sdk: docker
app_port: 8501
tags:
  - streamlit
pinned: false
short_description: Zero trust. Zero phishing. Zero compromises.
license: mit

ZeroPhish Gate - AI-Powered Phishing Detection System
Zero trust. Zero phishing. Zero compromises.

ZeroPhish Gate is a multilingual, AI-powered security assistant designed to detect and explain phishing, spam, and fraudulent content in emails, chats, or files. It combines traditional pattern-based techniques with semantic LLM reasoning to ensure both accuracy and human-friendly feedback.

Presentation + Demo Video: 8 minute Video
https://drive.google.com/file/d/1KXUa65tDNqBkxaN1tHW-zmxl3RFsl7y2/view?usp=sharing
Key Features
Hybrid AI Threat Detection
BERT-based pattern detection for known phishing traits
LLaMA (via Groq API) for intent recognition and semantic analysis
Retrieval-Augmented Generation (RAG) for reranked, context-rich results
Multilingual & Accessible
Supports 40+ languages
Text-to-speech output via gTTS for accessibility
User-Focused Output
Input via plain text or uploaded PDF/TXT files
Role-based actionable advice (e.g., for procurement, admin, finance)
Glossary with hover-over terms for easy learning
Downloadable security reports and summaries
Risk Scoring & Feedback
Visual badges with 5-tier risk level (0–100%)
Interactive threat analysis history
Audio summary of the scan for non-readers
Quick Start Guide
Paste or Upload:

Paste suspicious content OR upload a .pdf or .txt file.
Select Language & Role:

Choose your preferred language and organizational role.
Run Analysis:

Click "Analyze with AI" to evaluate for threats.
Review & Act:

View detailed analysis with glossary and tips.
Download a report or listen to a voice summary.
Report suspicious content to IT.
How It Works (Simplified)
You enter suspicious content or upload a document.
A BERT model checks for known phishing patterns.
LLaMA interprets tone and context using natural language.
Threat score is calculated, and advice is generated.
You get results with clear visuals, definitions, and audio if needed.
Technical Architecture
Architecture Diagram

Diagram shows pipeline from input → analysis → scoring → output

Architecture (Mermaid version)
Risk Scoring System
Score Range	Level	Color	Description
0–20	Safe	🟢	No threat detected
21–40	Minimal Suspicion	🟡	Minor concerns, safe to review
41–60	Needs Attention	🟠	Potential risk, review content
61–80	Likely Threat	🔴	High probability of phishing/spam
81–100	Severe Threat	⚫	Dangerous, report and avoid
Glossary (Sample Terms)
Hover over underlined terms in the app to learn more.

Local Installation
# Clone the repo
git clone https://huggingface.co/spaces/your-username/ZeroPhish-Gate
cd ZeroPhish-Gate

# Install dependencies
pip install -r requirements.txt

# Run the app (Gradio prototype)
python app.py

# OR run with Streamlit UI
streamlit run ui_app.py
Requirements
Python 3.8+

Required packages:

gradio
fitz  # PyMuPDF
gtts
transformers
python-dotenv
groq
Required API Keys
Make sure to set the following in your .env file or environment variables:

GROQ_API_KEY
HF_TOKEN
KAGGLE_USERNAME
KAGGLE_KEY
License
MIT License – see LICENSE for details.

Contributions
We welcome community contributions, issue reports, and feedback! Feel free to open a pull request or start a discussion.
