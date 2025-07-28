import os
import re
import gradio as gr
from datetime import datetime
import tempfile
import io
import base64

# Check for required dependencies
try:
    import fitz  # PyMuPDF

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not available, PDF support disabled")

try:
    from gtts import gTTS

    TTS_SUPPORT = True
except ImportError:
    TTS_SUPPORT = False
    print("Warning: gTTS not available, audio synthesis disabled")

try:
    from groq import Groq

    GROQ_SUPPORT = True
except ImportError:
    GROQ_SUPPORT = False
    print("Warning: Groq not available")

try:
    from transformers import pipeline

    HF_TRANSFORMERS_SUPPORT = True
except ImportError:
    HF_TRANSFORMERS_SUPPORT = False
    print("Warning: Transformers not available")

# ✅ Load secrets from environment with better error handling
groq_key = os.environ.get('GROQ_API_KEY')
hf_token = os.environ.get('HF_TOKEN')
kaggle_key = os.environ.get('KAGGLE_KEY')
kaggle_username = os.environ.get('KAGGLE_USERNAME')

# Ensure none of the required secrets are missing
if not all([groq_key, hf_token]):
    raise EnvironmentError("❌ One or more required API keys are missing from environment variables.")

# Initialize components with fallbacks
client = None
phishing_pipe = None

if GROQ_SUPPORT and groq_key:
    try:
        client = Groq(api_key=groq_key)
        print("✅ Groq client initialized")
    except Exception as e:
        print(f"❌ Groq initialization failed: {e}")

if HF_TRANSFORMERS_SUPPORT and hf_token:
    try:
        phishing_pipe = pipeline(
            "text-classification",
            model="ealvaradob/bert-finetuned-phishing",
            token=hf_token,
            return_all_scores=True
        )
        print("✅ Phishing detection model loaded")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        # Try alternative model
        try:
            phishing_pipe = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                return_all_scores=True
            )
            print("✅ Fallback model loaded")
        except Exception as e2:
            print(f"❌ Fallback model also failed: {e2}")

# Global variables
history_log = []
detailed_log = []

# 🎯 Role options
role_choices = ["Procurement", "Warehouse", "Admin", "Finance", "Logistics"]

# 🌍 Language options
language_choices = [
    "English", "Urdu", "Arabic", "French", "German", "Spanish", "Portuguese", "Hindi", "Turkish",
    "Bengali", "Russian", "Chinese", "Japanese", "Korean", "Swahili", "Indonesian", "Italian",
    "Dutch", "Polish", "Thai", "Vietnamese", "Romanian", "Persian", "Punjabi", "Greek", "Hebrew",
    "Malay", "Czech", "Danish", "Finnish", "Hungarian", "Norwegian", "Slovak", "Swedish", "Tamil",
    "Telugu", "Gujarati", "Marathi", "Pashto", "Serbian", "Croatian", "Ukrainian", "Bulgarian",
    "Filipino", "Sinhala", "Mongolian", "Kazakh", "Azerbaijani", "Nepali", "Malayalam"
]

# Glossary terms with tooltip
GLOSSARY = {
    "phishing": "Phishing is a scam where attackers trick you into revealing personal information.",
    "domain spoofing": "Domain spoofing is when someone fakes a legitimate website's address to deceive you.",
    "malware": "Malicious software designed to harm or exploit systems.",
    "spam": "Unwanted or unsolicited messages.",
    "tone": "The emotional character of the message."
}


def extract_text_from_file(file_obj):
    """Extract text from uploaded files with error handling"""
    if file_obj is None:
        return ""

    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        ext = file_path.split(".")[-1].lower()

        if ext == "pdf" and PDF_SUPPORT:
            doc = fitz.open(file_path)
            return "\n".join(page.get_text() for page in doc)
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            return f"Unsupported file type: {ext}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def rag_based_reranking(text, bert_analysis, language="English"):
    """
    RAG/Prompt-Based Reranking: Use LLaMA to reanalyze and improve BERT's classification
    This adds semantic analysis and intent understanding
    """
    try:
        # Create prompt for LLaMA semantic reanalysis
        reranking_prompt = [
            {
                "role": "system",
                "content": f"""
You are an expert cybersecurity analyst specializing in phishing detection.
Your job is to reanalyze email/message content using semantic understanding and intent analysis.
You have received a preliminary classification from a BERT model, but you need to provide a more accurate assessment using your understanding of:
- Social engineering tactics
- Urgency patterns
- Suspicious requests
- Context and intent
- Language patterns that indicate deception
Respond with your reanalysis in this exact format:
REANALYZED_THREAT_TYPE: [safe/spam/phishing/malware]
CONFIDENCE_LEVEL: [low/medium/high]
REASONING: [Brief explanation of your decision]
SEMANTIC_INDICATORS: [What semantic clues led to this conclusion]
"""
            },
            {
                "role": "user",
                "content": f"""
ORIGINAL MESSAGE TO ANALYZE:
"{text}"
BERT MODEL'S PRELIMINARY ANALYSIS:
- Classification: {bert_analysis.get('model_prediction', 'Unknown')}
- Threat Type: {bert_analysis.get('threat_type', 'unknown')}
- AI Confidence: {bert_analysis.get('ai_confidence_score', 0)}%
TASK: Does this message pose a phishing, spam, malware, or other security risk?
Use your semantic understanding to reanalyze this message. Consider:
1. Intent and context
2. Social engineering patterns
3. Urgency or pressure tactics
4. Suspicious requests (credentials, money, personal info)
5. Language patterns that suggest deception
6. Overall trustworthiness
Provide your reanalysis using the format specified above.
"""
            }
        ]

        # Get LLaMA's semantic reanalysis
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=reranking_prompt,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=500
        )

        llama_response = response.choices[0].message.content.strip()

        # Parse LLaMA's response
        reanalysis = parse_llama_reanalysis(llama_response)

        # Combine BERT and LLaMA insights
        final_analysis = combine_bert_llama_analysis(bert_analysis, reanalysis, text)

        return final_analysis

    except Exception as e:
        print(f"RAG Reranking failed: {e}")
        # Fallback to original BERT analysis
        bert_analysis['rag_error'] = str(e)
        return bert_analysis


def parse_llama_reanalysis(llama_response):
    """Parse LLaMA's structured response"""
    reanalysis = {
        'llama_threat_type': 'unknown',
        'llama_confidence': 'medium',
        'llama_reasoning': '',
        'semantic_indicators': ''
    }

    lines = llama_response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('REANALYZED_THREAT_TYPE:'):
            reanalysis['llama_threat_type'] = line.split(':', 1)[1].strip().lower()
        elif line.startswith('CONFIDENCE_LEVEL:'):
            reanalysis['llama_confidence'] = line.split(':', 1)[1].strip().lower()
        elif line.startswith('REASONING:'):
            reanalysis['llama_reasoning'] = line.split(':', 1)[1].strip()
        elif line.startswith('SEMANTIC_INDICATORS:'):
            reanalysis['semantic_indicators'] = line.split(':', 1)[1].strip()

    return reanalysis


def combine_bert_llama_analysis(bert_analysis, llama_reanalysis, original_text):
    """
    Combine BERT and LLaMA analysis using hybrid decision logic
    LLaMA's semantic understanding gets priority for final classification
    """

    # Get both predictions
    bert_threat = bert_analysis.get('threat_type', 'unknown')
    llama_threat = llama_reanalysis.get('llama_threat_type', 'unknown')
    llama_confidence = llama_reanalysis.get('llama_confidence', 'medium')

    # Hybrid decision logic
    if llama_confidence == 'high':
        # Trust LLaMA's high-confidence assessment
        final_threat_type = llama_threat
        decision_method = "LLaMA High Confidence"
    elif bert_threat == llama_threat:
        # Both models agree - high confidence in result
        final_threat_type = bert_threat
        decision_method = "BERT + LLaMA Agreement"
    elif llama_threat == 'safe' and bert_threat in ['spam', 'unknown']:
        # LLaMA says safe, BERT unsure - lean towards safe
        final_threat_type = 'safe'
        decision_method = "LLaMA Safety Override"
    elif llama_threat in ['phishing', 'malware'] and bert_threat != 'safe':
        # LLaMA detects serious threat - prioritize security
        final_threat_type = llama_threat
        decision_method = "LLaMA Threat Detection"
    else:
        # Default to BERT with LLaMA insights
        final_threat_type = bert_threat
        decision_method = "BERT with LLaMA Insights"

    # Create enhanced analysis combining both models
    enhanced_analysis = bert_analysis.copy()
    enhanced_analysis.update({
        'final_threat_type': final_threat_type,
        'bert_prediction': bert_threat,
        'llama_prediction': llama_threat,
        'llama_confidence': llama_confidence,
        'llama_reasoning': llama_reanalysis.get('llama_reasoning', ''),
        'semantic_indicators': llama_reanalysis.get('semantic_indicators', ''),
        'decision_method': decision_method,
        'hybrid_analysis': True
    })

    # Recalculate threat score based on final classification
    enhanced_analysis['threat_type'] = final_threat_type
    threat_score = calculate_threat_score(enhanced_analysis)
    enhanced_analysis['threat_score'] = threat_score

    return enhanced_analysis


def calculate_threat_score(hf_analysis):
    """
    Calculate threat score based on AI analysis results
    Returns score from 0-100 where higher means more dangerous
    """
    threat_type = hf_analysis.get('threat_type', 'unknown')
    confidence_percentage = hf_analysis.get('ai_confidence_score', 0)

    if threat_type == 'safe':
        # For safe messages, use INVERSE of confidence
        # High confidence in "safe" = Low threat score
        threat_score = max(0, min(20, (100 - confidence_percentage) * 0.2))

    elif threat_type == 'spam':
        # For spam, map confidence to 21-40% range
        threat_score = 21 + (confidence_percentage * 0.19)

    elif threat_type == 'phishing':
        # For phishing, map confidence to 61-80% range
        threat_score = 61 + (confidence_percentage * 0.19)

    elif threat_type == 'malware':
        # For malware, map confidence to 81-100% range
        threat_score = 81 + (confidence_percentage * 0.19)

    else:
        # For unknown threats, use moderate risk
        threat_score = 41 + (confidence_percentage * 0.19)

    # Ensure score stays within bounds
    threat_score = round(min(100, max(0, threat_score)), 1)

    # Additional safety check for very short, innocent messages
    text = hf_analysis.get('raw_text', '')
    if len(text.strip()) <= 10 and threat_type == 'safe':
        threat_score = min(threat_score, 10.0)

    return threat_score


def analyze_with_huggingface(text):
    """
    First stage: Analyze message using Hugging Face BERT model
    Returns detailed technical analysis for LLaMA to interpret
    FIXED VERSION: Properly handles safe messages like "HI"
    """
    try:
        # Get prediction from Hugging Face model
        result = phishing_pipe(text)

        # Extract prediction details
        prediction = result[0]
        label = prediction['label']
        confidence_score = prediction['score']

        # Convert to percentage
        confidence_percentage = round(confidence_score * 100, 2)

        # Map labels to threat types (adjust based on your model's labels)
        threat_mapping = {
            'PHISHING': 'phishing',
            'LEGITIMATE': 'safe',
            'SPAM': 'spam',
            'MALWARE': 'malware'
        }

        threat_type = threat_mapping.get(label.upper(), 'unknown')

        # FIXED LOGIC: Calculate threat score based on what the model actually detected
        if threat_type == 'safe':
            # For safe messages, use INVERSE of confidence
            # High confidence in "safe" = Low threat score
            threat_score = max(0, min(20, (100 - confidence_percentage) * 0.2))
            threat_level = "Safe"

        elif threat_type == 'spam':
            # For spam, map confidence to 21-40% range
            threat_score = 21 + (confidence_percentage * 0.19)
            threat_level = "Minimal Suspicion"

        elif threat_type == 'phishing':
            # For phishing, map confidence to 61-80% range
            threat_score = 61 + (confidence_percentage * 0.19)
            threat_level = "Likely Threat"

        elif threat_type == 'malware':
            # For malware, map confidence to 81-100% range
            threat_score = 81 + (confidence_percentage * 0.19)
            threat_level = "Severe Threat"

        else:
            # For unknown threats, use moderate risk
            threat_score = 41 + (confidence_percentage * 0.19)
            threat_level = "Needs Attention"

        # Ensure score stays within bounds
        threat_score = round(min(100, max(0, threat_score)), 1)

        # Additional safety check for very short, innocent messages
        if len(text.strip()) <= 10 and threat_type == 'safe':
            # For very short messages classified as safe, ensure low threat score
            threat_score = min(threat_score, 10.0)
            threat_level = "Safe"

        # Create technical analysis summary for LLaMA
        technical_analysis = {
            'model_prediction': label,
            'ai_confidence_score': confidence_percentage,
            'threat_type': threat_type,
            'threat_score': threat_score,
            'threat_level': threat_level,
            'raw_text': text[:500]
        }

        return technical_analysis

    except Exception as e:
        # Fallback analysis if Hugging Face model fails
        return {
            'model_prediction': 'UNKNOWN',
            'ai_confidence_score': 0,
            'threat_type': 'unknown',
            'threat_score': 50.0,
            'threat_level': 'Needs Attention',
            'raw_text': text[:500],
            'error': str(e)
        }


def build_enhanced_prompt_messages(hf_analysis, language="English", role="Admin"):
    """
    Build prompt that includes both BERT and LLaMA reanalysis for final interpretation
    """
    # Check if hybrid analysis was performed
    if hf_analysis.get('hybrid_analysis', False):
        technical_data = f"""
HYBRID AI ANALYSIS RESULTS:
- BERT Model Prediction: {hf_analysis.get('bert_prediction', 'Unknown')}
- LLaMA Semantic Analysis: {hf_analysis.get('llama_prediction', 'Unknown')}
- Final Classification: {hf_analysis['final_threat_type']}
- Decision Method: {hf_analysis.get('decision_method', 'Standard')}
- LLaMA Confidence: {hf_analysis.get('llama_confidence', 'medium')}
- Threat Score: {hf_analysis['threat_score']}% (0-100, higher = more dangerous)
- LLaMA Reasoning: {hf_analysis.get('llama_reasoning', 'N/A')}
- Semantic Indicators: {hf_analysis.get('semantic_indicators', 'N/A')}
- Original Message: "{hf_analysis['raw_text']}"
"""
    else:
        technical_data = f"""
STANDARD AI ANALYSIS:
- Model Prediction: {hf_analysis['model_prediction']}
- Detected Threat Type: {hf_analysis['threat_type']}
- Threat Score: {hf_analysis['threat_score']}% (0-100, higher = more dangerous)
- Original Message: "{hf_analysis['raw_text']}"
"""

    if 'error' in hf_analysis or 'rag_error' in hf_analysis:
        error_msg = hf_analysis.get('error', hf_analysis.get('rag_error', ''))
        technical_data += f"\n- Analysis Note: {error_msg}"

    return [
        {
            "role": "system",
            "content": f"""
You are a friendly cybersecurity assistant helping employees in the supply chain industry.
You have received results from a hybrid AI analysis system that combines:
1. BERT model for technical pattern detection
2. LLaMA model for semantic understanding and intent analysis
Your job is to explain these results in SIMPLE, NON-TECHNICAL language that anyone can understand.
Guidelines:
- Use everyday words instead of technical jargon
- The threat score ranges from 0-100 where higher numbers mean more dangerous
- Explain both what the computers found AND why it matters
- Give clear, practical advice for a {role} employee
- If there's disagreement between models, explain what that means
Always respond completely in {language} only.
Make it sound like you're talking to a friend, not giving a technical report.
"""
        },
        {
            "role": "user",
            "content": f"""
Please analyze this hybrid security check and explain it in simple terms:
{technical_data}
Respond in this format using everyday language:
1. Tone: (How does the message sound? Pushy, friendly, normal, etc.)
2. Threat Type: (What kind of danger is this? Safe message, scam attempt, spam, etc.)
3. Threat Score: (Show the danger level number from 0-100)
4. AI Analysis Summary: (What did both computer systems find? Did they agree?)
5. Simple Explanation (in {language}): (Explain in plain words why this is safe or dangerous)
6. What should you do as a {role} worker (in {language}): (Clear, simple steps)
7. Why the computers flagged this: (Explain what the AI systems noticed)
8. Detailed Advisory (in {language}): (Comprehensive guidance and precautions)
"""
        }
    ]


def get_threat_level_display(threat_score):
    """
    Get color-coded threat level display based on corrected 5-level system
    """
    if threat_score <= 20:
        return "🟢 SAFE - No threat detected"
    elif threat_score <= 40:
        return "🟡 MINIMAL SUSPICION - Minor concerns"
    elif threat_score <= 60:
        return "🟠 NEEDS ATTENTION - Requires careful review"
    elif threat_score <= 80:
        return "🔴 LIKELY THREAT - High probability of danger"
    else:
        return "⚫ SEVERE THREAT - Immediate action required"


def generate_text_report(analysis_data, hf_analysis, input_text):
    """
    Generate a structured text report that can be downloaded as a .txt file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
================================================================================
                    ZEROPHISH GATE - SECURITY ANALYSIS REPORT
================================================================================
Analysis Date: {timestamp}
Generated by: ZeroPhish Gate AI Security System
================================================================================
ANALYZED MESSAGE
================================================================================
{input_text}
================================================================================
THREAT ASSESSMENT SUMMARY
================================================================================
AI Detection:     {hf_analysis.get('model_prediction', 'Unknown')}
Message Type:     {hf_analysis.get('threat_type', 'Unknown').title()}
Threat Score:     {hf_analysis.get('threat_score', 'N/A')}%
================================================================================
DETAILED ANALYSIS
================================================================================
"""

    # Parse detailed analysis sections
    sections = {}
    current_section = ""
    lines = analysis_data.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith(('1. Tone:', '2. Threat Type:', '3. Threat Score:', '4. Simple Explanation',
                            '5. What should you do', '6. Why the computer', '7. Detailed Advisory')):
            current_section = line
            sections[current_section] = ""
        elif current_section and line and not line.startswith('🤖'):
            sections[current_section] += line + " "

    # Add detailed analysis sections
    section_titles = [
        ("1. Tone:", "MESSAGE TONE ANALYSIS"),
        ("2. Threat Type:", "THREAT CLASSIFICATION"),
        ("3. Threat Score:", "THREAT SCORE ASSESSMENT"),
        ("4. Simple Explanation", "DETAILED EXPLANATION"),
        ("5. What should you do", "RECOMMENDED ACTIONS"),
        ("6. Why the computer", "AI DETECTION REASONING"),
        ("7. Detailed Advisory", "COMPREHENSIVE ADVISORY")
    ]

    for section_key, section_title in section_titles:
        content_text = ""
        for key, value in sections.items():
            if key.startswith(section_key):
                content_text = value.strip()
                break

        if content_text:
            report += f"""
{section_title}
{'-' * len(section_title)}
{content_text}
"""

    # Footer
    report += f"""
================================================================================
REPORT FOOTER
================================================================================
This report was generated by ZeroPhish Gate AI Security System.
For support or questions, contact your IT security team.
Report ID: ZPG-{timestamp.replace(' ', 'T').replace(':', '-')}
Analysis completed at: {timestamp}
================================================================================
"""

    return report


def generate_downloadable_report(analysis_data, hf_analysis, input_text):
    """
    Generate a downloadable report file
    """
    # Create text report
    text_report = generate_text_report(analysis_data, hf_analysis, input_text)

    # Create downloadable file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"zerophish_security_report_{timestamp}.txt"

    # Create a temporary file for download
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(text_report)
        temp_path = f.name

    return temp_path


def add_visual_badges(text):
    tags = []
    if "urgent" in text.lower():
        tags.append("🔴 Urgent Tone Detected")
    if "suspicious" in text.lower() and "domain" in text.lower():
        tags.append("🗘 Suspicious Domain")
    if "safe" in text.lower() or "threat type: safe" in text.lower():
        tags.append("🟩 Clean")
    if "ai model" in text.lower():
        tags.append("🤖 AI-Enhanced Analysis")
    if tags:
        return text + "\n\n🚨 Visual Tags:\n" + "\n".join(tags)
    return text


def apply_glossary_tooltips(text):
    """Apply HTML tooltips for glossary terms"""
    # First, convert newlines to HTML breaks
    text = text.replace('\n', '<br>')

    # Apply tooltips to glossary terms
    for term, definition in GLOSSARY.items():
        pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
        # Create HTML span with title attribute for tooltip
        tooltip_html = f'<span title="{definition}" style="font-weight: bold; text-decoration: underline; cursor: help; color: #0066cc;">{term}</span>'
        text = pattern.sub(tooltip_html, text)

    # Handle markdown-style bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    # Wrap in a simple div
    html_output = f'<div style="padding: 10px; line-height: 1.5; font-size: 14px;">{text}</div>'

    return html_output


def synthesize_audio(text, language="English"):
    if not text.strip():
        return None

    try:
        # Language code mapping (expand as needed)
        lang_map = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "hindi": "hi",
            "arabic": "ar",
            "urdu": "ur"  # gTTS supports Urdu as 'ur'
        }

        # Get language code (default to English if not found)
        lang_code = lang_map.get(language.lower(), "en")

        # Generate speech with proper language handling
        tts = gTTS(text=text[:200], lang=lang_code, slow=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tts.save(tmp_path)
            return tmp_path

    except Exception as e:
        print(f"Audio synthesis error: {e}")
        return None


def new_chat():
    """Reset all fields for a new chat session"""
    return (
        "",  # text_input
        None,  # file_input
        "",  # output
        gr.update(visible=False),  # report_btn
        gr.update(visible=False),  # ignore_btn
        gr.update(visible=False),  # report_msg
        None,  # audio_output
        None,  # report_file
    )


def analyze_message_interface(text_input, uploaded_file, language, role):
    file_text = extract_text_from_file(uploaded_file) if uploaded_file else ""
    text_input = text_input.strip()
    file_text = file_text.strip()

    if not text_input and not file_text:
        return "❌ No input provided via text or file.", gr.update(visible=False), history_log, gr.update(choices=[],
                                                                                                         value=None), "", None, gr.update(
            visible=False), gr.update(visible=False), None

    combined_input = f"User message:\n{text_input}\n\nAttached file content:\n{file_text}" if text_input and file_text else (
                text_input or file_text)

    # STAGE 1: Hugging Face BERT Analysis
    print("🤖 Stage 1: Running Hugging Face BERT analysis...")
    bert_analysis = analyze_with_huggingface(combined_input)

    # STAGE 1.5: RAG-Based Reranking with LLaMA Semantic Analysis
    print("🧠 Stage 1.5: RAG-based reranking with LLaMA semantic analysis...")
    hf_analysis = rag_based_reranking(combined_input, bert_analysis, language)

    # Calculate final threat score
    threat_score = calculate_threat_score(hf_analysis)
    hf_analysis['threat_score'] = threat_score

    # Stage 2: Final LLaMA Interpretation
    print("🧠 Stage 2: Final LLaMA interpretation of hybrid analysis...")
    messages = build_enhanced_prompt_messages(hf_analysis, language, role)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )
    result = response.choices[0].message.content.strip()

    # In your analyze_message_interface function:
    audio_path = synthesize_audio(result, language)  # Pass the same language selected in UI

    # Add hybrid analysis summary to the result
    if hf_analysis.get('hybrid_analysis', False):
        result += f"\n\n🤖 **Hybrid AI Analysis Summary:**\n"
        result += f"- BERT Detection: {hf_analysis.get('bert_prediction', 'Unknown').title()}\n"
        result += f"- LLaMA Reanalysis: {hf_analysis.get('llama_prediction', 'Unknown').title()}\n"
        result += f"- Final Decision: {hf_analysis['final_threat_type'].title()}\n"
        result += f"- Method: {hf_analysis.get('decision_method', 'Standard')}\n"
        result += f"- Threat Score: {hf_analysis['threat_score']}%"
    else:
        result += f"\n\n🤖 **Computer Analysis Summary:**\n- Detection: {hf_analysis['model_prediction']}\n- Message Type: {hf_analysis['threat_type'].title()}\n- Threat Score: {hf_analysis['threat_score']}%"

    result_with_badges = add_visual_badges(result)
    result_with_tooltips = apply_glossary_tooltips(result_with_badges)

    # Determine if threat based on final analysis
    final_threat_type = hf_analysis.get('final_threat_type', hf_analysis.get('threat_type', 'unknown'))
    is_threat = hf_analysis['threat_score'] > 20

    # Extract information for logging
    threat_score_str = f"{hf_analysis['threat_score']}%"
    status = "Safe" if final_threat_type == 'safe' else "Review"

    history_log.append([
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        combined_input[:40] + "...",
        threat_score_str,
        final_threat_type.title(),
        status
    ])

    # Store data for detailed view
    detailed_log.append({
        "full_input": combined_input,
        "full_result": result_with_badges,
        "hf_analysis": hf_analysis,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    audio_path = synthesize_audio(result, language)

    # Generate downloadable report
    report_file = generate_downloadable_report(result_with_badges, hf_analysis, combined_input)

    return (
        result_with_tooltips,  # output as plain text
        gr.update(visible=is_threat),  # report_btn
        history_log,
        gr.update(choices=[str(i) for i in range(len(detailed_log))], value=None),
        "",  # full_view
        audio_path,
        gr.update(visible=is_threat),  # report_msg
        gr.update(visible=is_threat),  # ignore_btn
        report_file  # report_file for download
    )


def view_full_report(index):
    try:
        idx = int(index)
        record = detailed_log[idx]
        hf_data = record['hf_analysis']

        report = f"📅 **Timestamp:** {record['timestamp']}\n\n"
        report += f"📝 **Input Message:**\n{record['full_input']}\n\n"
        report += f"🤖 **Computer Security Check:**\n"
        report += f"- What AI Found: {hf_data['model_prediction']}\n"
        report += f"- Message Type: {hf_data['threat_type']}\n"
        report += f"- Threat Score: {hf_data.get('threat_score', 'N/A')}%\n"
        if 'error' in hf_data:
            report += f"- Note: {hf_data['error']}\n"
        report += f"\n📜 **Detailed Analysis:**\n{record['full_result']}"

        return report
    except:
        return "❌ Invalid selection"


def report_to_it(language):
    english_msg = "✅ Your request has been forwarded to the concerning department..."
    if history_log:
        history_log[-1][4] = "Reported"

    if language.lower() == "english":
        return english_msg, history_log

    prompt = [{
        "role": "user",
        "content": f'Translate this message into {language} and include the English version in brackets:\n\n"{english_msg}"'
    }]
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=prompt,
        temperature=0.2,
        max_tokens=250
    )
    output = response.choices[0].message.content.strip()

    match = re.search(r'([^\[]+)(\[[^\]]+\])', output)
    if match:
        return f"{match.group(1).strip()}\n{match.group(2).strip()}", history_log
    else:
        return f"{output}\n[{english_msg}]", history_log


def ignore_latest():
    if history_log:
        history_log[-1][4] = "Ignored"
    return history_log


def clear_history():
    history_log.clear()
    detailed_log.clear()
    return [], [], "", gr.update(visible=False)


def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
            title="ZeroPhish Gate - Phishing Detection",
            theme=gr.themes.Soft(),
            css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .analysis-output { padding: 15px; border-radius: 10px; }
        """
    ) as demo:

        gr.HTML("""
        <div class="main-header">
            <h1>🛡️ ZeroPhish Gate</h1>
            <h3>AI-Powered Phishing & Threat Detection</h3>
            <p>Analyze messages, emails, and documents for potential security threats</p>
        </div>
        """)

        # System status
        status_msg = "🟢 System Ready"
        if not (GROQ_SUPPORT and groq_key):
            status_msg += " (Advanced AI disabled)"
        if not phishing_pipe:
            status_msg += " (Using basic detection)"

        gr.Markdown(f"**Status:** {status_msg}")

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="📝 Message to Analyze",
                    placeholder="Paste suspicious email, SMS, or message here...",
                    lines=5
                )

                file_input = gr.File(
                    label="📎 Upload File (Optional)",
                    file_types=[".txt", ".pdf"] if PDF_SUPPORT else [".txt"]
                )

            with gr.Column(scale=1):
                role = gr.Dropdown(
                    label="👤 Your Role",
                    choices=role_choices,
                    value="Admin"
                )

                language = gr.Dropdown(
                    label="🌐 Language",
                    choices=language_choices,
                    value="English"
                )

        analyze_btn = gr.Button(
            "🔍 Analyze Message",
            variant="primary",
            size="lg"
        )

        new_chat_btn = gr.Button("🆕 New Chat", variant="secondary")

        with gr.Row():
            output = gr.HTML(
                label="📊 Analysis Results",
                elem_classes=["analysis-output"]
            )

        with gr.Row():
            report_btn = gr.Button("🚨 Report to IT", visible=False, variant="stop")
            ignore_btn = gr.Button("🙊 Ignore Message", visible=False)

        report_msg = gr.Textbox(label="📣 IT Confirmation", visible=False, interactive=False)

        with gr.Row():
            audio_output = gr.Audio(label="🔊 Voice Output (Click to Play)", interactive=False, autoplay=False)
            report_file = gr.File(label="📄 Download Security Report", interactive=False)

        with gr.Accordion("📜 Risk History & Detailed Reports", open=False):
            history_table = gr.Dataframe(
                headers=["Time", "Preview", "Threat Score", "Type", "Status"],
                label="📜 Risk History Log",
                interactive=False,
                wrap=True
            )
            clear_btn = gr.Button("🧹 Clear History")

            selected_idx = gr.Dropdown(label="📂 Select Report to View", choices=[], interactive=True)
            full_view = gr.Textbox(label="🔍 Detailed Analysis", lines=12, interactive=False)

        with gr.Accordion("ℹ️ Help & Information", open=False):
            gr.Markdown("""
            ### How to Use
            1. **Paste or type** the suspicious message in the text box
            2. **Upload a file** (PDF or TXT) if needed
            3. **Select your role** for personalized advice
            4. **Click Analyze** to get results

            ### Threat Types
            - 🟢 **Safe**: No threats detected
            - 🟡 **Spam**: Unwanted promotional content
            - 🟠 **Suspicious**: Potentially harmful content
            - 🔴 **Phishing**: Attempts to steal information
            - 🔴 **Malware**: Malicious software threats

            ### Tips
            - Always verify suspicious requests through official channels
            - Never click links or download attachments from unknown senders
            - When in doubt, contact your IT security team
            """)

        with gr.Accordion("📚 Glossary Help", open=False):
            gr.Markdown("""
            **Hover over underlined blue terms in the analysis to see their definitions:**
            - **Phishing**: A type of online scam where attackers trick you into giving away personal information
            - **Domain Spoofing**: When a fake website mimics a trusted one by using a similar-looking web address
            - **Malware**: Software designed to harm or gain unauthorized access to your device or data
            - **Spam**: Unwanted or unsolicited messages, usually advertisements or scams
            - **Tone**: The emotional tone in a message, like being urgent or friendly
            """)

        with gr.Accordion("🤖 AI Pipeline Info", open=False):
            gr.Markdown("""
            **Three-Stage Hybrid Analysis Pipeline:**
            1. **Stage 1 - BERT Model:** Technical phishing pattern detection
            2. **Stage 1.5 - RAG Reranking:** LLaMA semantic reanalysis for intent understanding
            3. **Stage 2 - Final Interpretation:** User-friendly explanation generation
            **RAG-Based Reranking Benefits:**
            ✅ **Semantic Understanding:** LLaMA analyzes intent and context, not just patterns
            ✅ **Social Engineering Detection:** Better detection of psychological manipulation
            ✅ **Hybrid Decision Making:** Combines pattern matching with contextual analysis
            ✅ **Reduced False Positives:** More accurate classification of legitimate messages
            **How It Works:**
            - BERT identifies technical patterns (suspicious links, keywords, etc.)
            - LLaMA reanalyzes for social engineering, urgency, and intent
            - System combines both analyses for final classification
            - Prioritizes safety while reducing false alarms
            **Message Classification:**
            - Safe: Normal, legitimate messages
            - Spam: Unwanted promotional content
            - Phishing: Attempts to steal personal information
            - Malware: Messages with malicious attachments or links
            """)

        # Event handlers
        analyze_btn.click(
            fn=analyze_message_interface,
            inputs=[text_input, file_input, language, role],
            outputs=[output, report_btn, history_table, selected_idx, full_view, audio_output, report_msg, ignore_btn,
                     report_file]
        )

        new_chat_btn.click(
            fn=new_chat,
            inputs=[],
            outputs=[text_input, file_input, output, report_btn, ignore_btn, report_msg, audio_output, report_file]
        )

        selected_idx.change(fn=view_full_report, inputs=[selected_idx], outputs=[full_view])
        report_btn.click(fn=report_to_it, inputs=[language], outputs=[report_msg, history_table])
        report_btn.click(lambda: gr.update(visible=True), outputs=report_msg)
        ignore_btn.click(fn=ignore_latest, outputs=[history_table])
        clear_btn.click(fn=clear_history, outputs=[history_table, selected_idx, full_view, report_msg])

    return demo


# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )