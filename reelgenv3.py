import os
import json
import random
import sqlite3
import pandas as pd
import hashlib
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import streamlit as st
from dotenv import load_dotenv
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# -----------------------------
# ENV / Konfiguration
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Session State für History und Analytics
if "generated_history" not in st.session_state:
    st.session_state.generated_history = []
if "analytics_tracker" not in st.session_state:
    st.session_state.analytics_tracker = {
        "total_generations": 0,
        "total_ideas": 0,
        "avg_viral_score": 0,
        "topic_popularity": {},
        "tone_preferences": {},
        "generation_times": []
    }

# -----------------------------
# Datenbank-Integration
# -----------------------------
@contextmanager
def get_db_connection():
    """Context manager für Datenbankverbindung"""
    conn = sqlite3.connect('reel_generator.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialisiert die Datenbank mit benötigten Tabellen"""
    with get_db_connection() as conn:
        # Tabelle für Generierungen
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                parameters_hash TEXT,
                parameters TEXT,
                ideas TEXT,
                avg_viral_score REAL,
                user_rating INTEGER DEFAULT 0,
                feedback TEXT
            )
        """)
        
        # Tabelle für Content Templates
        conn.execute("""
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                category TEXT,
                structure TEXT,
                hook_formats TEXT,
                example TEXT,
                usage_count INTEGER DEFAULT 0
            )
        """)
        
        # Tabelle für Feedback
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER,
                rating INTEGER,
                comments TEXT,
                timestamp TEXT,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            )
        """)
        
        conn.commit()
    
    # Standard-Templates einfügen
    insert_default_templates()

def insert_default_templates():
    """Fügt vordefinierte Templates in die Datenbank ein"""
    templates = [
        {
            "name": "Bildendes 3-2-1 Format",
            "category": "Educational",
            "structure": "3 Fakten → 2 Fehler → 1 Lösung",
            "hook_formats": "Statistik, Frage, Überraschung",
            "example": "3 Anzeichen von Gaslighting, 2 häufige Fehler, 1 Weg raus",
            "usage_count": 0
        },
        {
            "name": "Storytelling Arc",
            "category": "Storytelling",
            "structure": "Setup → Konflikt → Auflösung → Lektion",
            "hook_formats": "Persönliche Geschichte, Emotionaler Einstieg",
            "example": "Wie ich aus einer toxischen Beziehung fand...",
            "usage_count": 0
        },
        {
            "name": "Quick Tips",
            "category": "Tips",
            "structure": "Tipp 1 → Tipp 2 → Tipp 3 → Zusammenfassung",
            "hook_formats": "Nummerierte Liste, Versprechen",
            "example": "5 Boundary-Setting Tipps in 30 Sekunden",
            "usage_count": 0
        }
    ]
    
    with get_db_connection() as conn:
        for template in templates:
            conn.execute("""
                INSERT OR IGNORE INTO templates (name, category, structure, hook_formats, example, usage_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (template["name"], template["category"], template["structure"], 
                  template["hook_formats"], template["example"], template["usage_count"]))
        conn.commit()

# -----------------------------
# Content Templates
# -----------------------------
CONTENT_TEMPLATES = {
    "educational": {
        "name": "Bildend",
        "structure": "Problem → Symptom → Lösung → Action",
        "hook_formats": ["Frage", "Statistik", "Persönliche Erfahrung"],
        "length": "30-45 Sekunden",
        "cta_style": "Wissensvermittlung",
        "description": "Erklärt komplexe Themen einfach"
    },
    "storytelling": {
        "name": "Storytelling",
        "structure": "Setup → Konflikt → Lösung → Moral",
        "hook_formats": ["Emotionaler Einstieg", "Überraschungsmoment", "Rhetorische Frage"],
        "length": "45-60 Sekunden",
        "cta_style": "Community-Engagement",
        "description": "Persönliche Geschichten mit Lerneffekt"
    },
    "quick_tips": {
        "name": "Quick Tips",
        "structure": "Tip 1 → Tip 2 → Tip 3 → Zusammenfassung",
        "hook_formats": ["Nummerierte Liste", "Versprechen", "Provokative These"],
        "length": "15-30 Sekunden",
        "cta_style": "Save/Bookmark",
        "description": "Kurze, umsetzbare Ratschläge"
    },
    "myth_busting": {
        "name": "Myth Busting",
        "structure": "Mythos → Faktencheck → Wahrheit → Takeaways",
        "hook_formats": ["Überraschender Fakt", "Gegensatz", "Klarstellung"],
        "length": "30-45 Sekunden",
        "cta_style": "Teilen",
        "description": "Räumt mit Missverständnissen auf"
    }
}

# -----------------------------
# Analytics Tracker
# -----------------------------
class AnalyticsTracker:
    """Verfolgt Nutzungsstatistiken und Performance"""
    
    def __init__(self):
        self.metrics = {
            "total_generations": 0,
            "total_ideas": 0,
            "total_tokens": 0,
            "avg_viral_score": 0,
            "topic_popularity": {},
            "tone_preferences": {},
            "generation_times": [],
            "user_ratings": []
        }
    
    def track_generation(self, ideas: List[Dict], params: Dict, generation_time: float):
        """Verfolgt eine neue Generierung"""
        self.metrics["total_generations"] += 1
        self.metrics["total_ideas"] += len(ideas)
        self.metrics["generation_times"].append(generation_time)
        
        # Themen-Popularität
        topic = params.get("topic", "Unknown")
        self.metrics["topic_popularity"][topic] = self.metrics["topic_popularity"].get(topic, 0) + 1
        
        # Ton-Präferenzen
        tone = params.get("tone", "Unknown")
        self.metrics["tone_preferences"][tone] = self.metrics["tone_preferences"].get(tone, 0) + 1
        
        # Durchschnittlicher Viral Score
        if ideas:
            avg_score = sum(idea.get("viral_score", 0) for idea in ideas) / len(ideas)
            self.metrics["avg_viral_score"] = (
                (self.metrics["avg_viral_score"] * (self.metrics["total_generations"] - 1) + avg_score) 
                / self.metrics["total_generations"]
            )
    
    def track_rating(self, rating: int):
        """Verfolgt User-Bewertungen"""
        self.metrics["user_ratings"].append(rating)
    
    def get_insights(self) -> Dict:
        """Gibt analysierte Insights zurück"""
        if not self.metrics["generation_times"]:
            return {"status": "No data yet"}
        
        return {
            "total_generations": self.metrics["total_generations"],
            "total_ideas": self.metrics["total_ideas"],
            "avg_generation_time": sum(self.metrics["generation_times"]) / len(self.metrics["generation_times"]),
            "avg_viral_score": self.metrics["avg_viral_score"],
            "top_topics": sorted(self.metrics["topic_popularity"].items(), key=lambda x: x[1], reverse=True)[:5],
            "preferred_tones": sorted(self.metrics["tone_preferences"].items(), key=lambda x: x[1], reverse=True)[:3],
            "avg_user_rating": sum(self.metrics["user_ratings"]) / len(self.metrics["user_ratings"]) if self.metrics["user_ratings"] else 0
        }

# -----------------------------
# Quality Control System
# -----------------------------
def quality_check_idea(idea: Dict) -> Dict:
    """Prüft eine Idee auf Qualitätsmerkmale"""
    checks = {
        'has_title': bool(idea.get('title')),
        'title_length': 10 <= len(idea.get('title', '')) <= 80,
        'has_hooks': len(idea.get('hooks', [])) >= 2,
        'caption_length': 50 <= len(idea.get('caption', '')) <= 2200,
        'has_hashtags': 5 <= len(idea.get('hashtags', [])) <= 20,
        'has_cta': bool(idea.get('cta')),
        'viral_score_range': 0 <= idea.get('viral_score', 0) <= 100,
        'has_visual_structure': bool(idea.get('visual_structure')),
        'has_posting_time': bool(idea.get('best_posting_time'))
    }
    
    quality_score = sum(checks.values()) / len(checks) * 100
    quality_issues = [k for k, v in checks.items() if not v]
    
    idea['quality_score'] = round(quality_score, 1)
    idea['quality_issues'] = quality_issues
    idea['quality_status'] = 'Excellent' if quality_score >= 90 else 'Good' if quality_score >= 70 else 'Needs Improvement'
    
    return idea

def batch_quality_check(ideas: List[Dict]) -> Dict:
    """Prüft eine Batch von Ideen"""
    checked_ideas = [quality_check_idea(idea) for idea in ideas]
    
    avg_quality = sum(idea['quality_score'] for idea in checked_ideas) / len(checked_ideas)
    common_issues = {}
    
    for idea in checked_ideas:
        for issue in idea['quality_issues']:
            common_issues[issue] = common_issues.get(issue, 0) + 1
    
    return {
        'checked_ideas': checked_ideas,
        'avg_quality_score': round(avg_quality, 1),
        'common_issues': sorted(common_issues.items(), key=lambda x: x[1], reverse=True),
        'excellent_count': sum(1 for idea in checked_ideas if idea['quality_status'] == 'Excellent'),
        'improvement_count': sum(1 for idea in checked_ideas if idea['quality_status'] == 'Needs Improvement')
    }

# -----------------------------
# Erweiterte OpenAI Funktionen
# -----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(client, messages, **kwargs):
    """OpenAI Call mit Retry-Logic"""
    try:
        return client.chat.completions.create(messages=messages, **kwargs)
    except openai.RateLimitError:
        st.warning("⚠️ Rate Limit erreicht. Warte 10 Sekunden...")
        time.sleep(10)
        raise
    except openai.APITimeoutError:
        st.warning("⚠️ Timeout. Versuche erneut...")
        raise
    except Exception as e:
        st.error(f"⚠️ API Fehler: {e}")
        raise

def create_parameters_hash(params: Dict) -> str:
    """Erstellt einen Hash für Caching-Zwecke"""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()

def check_cached_generation(params_hash: str) -> Optional[List[Dict]]:
    """Überprüft, ob diese Parameter schon generiert wurden"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT ideas FROM generations WHERE parameters_hash = ? LIMIT 1",
                (params_hash,)
            )
            result = cursor.fetchone()
            if result:
                return json.loads(result['ideas'])
    except Exception:
        pass
    return None

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Instagram Reel Generator PRO - Narzissmus Aufklärung",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CSS mit System-abhängigem Dark Mode
# -----------------------------
st.markdown("""<style>
    /* Light Mode (Standard) */
    .stApp { 
        background: linear-gradient(135deg, #f8f4f0 0%, #f0ebe5 100%); 
        transition: background 0.3s ease;
    }
    
    .content-card{
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .content-card:hover{
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .warning-box{
        background: rgba(255,193,7,0.1);
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        color: #856404;
    }
    
    .viral-badge-high{
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; padding: 8px 16px; border-radius: 20px;
        font-weight: 700; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .viral-badge-medium{
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white; padding: 8px 16px; border-radius: 20px;
        font-weight: 700; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .viral-badge-low{
        background: linear-gradient(135deg, #27ae60, #229954);
        color: white; padding: 8px 16px; border-radius: 20px;
        font-weight: 700; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .quality-badge-excellent{
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white; padding: 6px 12px; border-radius: 15px;
        font-weight: 600; font-size: 12px;
    }
    .quality-badge-good{
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white; padding: 6px 12px; border-radius: 15px;
        font-weight: 600; font-size: 12px;
    }
    .quality-badge-improve{
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; padding: 6px 12px; border-radius: 15px;
        font-weight: 600; font-size: 12px;
    }
    
    .main-title{
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
        transition: color 0.3s ease;
    }
    .subtitle{
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2em;
        margin-bottom: 30px;
        transition: color 0.3s ease;
    }
    
    /* Template Cards */
    .template-card{
        background: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border: 2px solid #667eea;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .template-card:hover{
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-3px);
    }
    .template-card.selected{
        background: rgba(102, 126, 234, 0.2);
        border-color: #764ba2;
    }
    
    /* Stats Cards */
    .stats-card{
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
        border-left: 4px solid #667eea;
    }
    
    /* Dark Mode (System-abhängig) */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #0b0f14 0%, #121826 100%) !important;
        }
        
        /* Globale Text-Farben */
        html, body, [class*="css"], .stMarkdown, .stText, .stCaption, 
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stMarkdownContainer"] span {
            color: #e6e6e6 !important;
        }
        
        /* Content Cards */
        .content-card{
            background: rgba(20,24,33,0.92) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.45) !important;
        }
        .content-card:hover{
            box-shadow: 0 12px 40px rgba(0,0,0,0.6) !important;
        }
        
        /* Titles */
        .main-title{ 
            color: #f1f5f9 !important; 
        }
        .subtitle{ 
            color: #b7c0cc !important; 
        }
        
        /* Metric Cards - leicht angepasst für Dark Mode */
        .metric-card{
            background: linear-gradient(135deg, #5a6fd8 0%, #6b4d96 100%) !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
        }
        
        /* Warning Box */
        .warning-box{
            background: rgba(255,193,7,0.15) !important;
            border-left: 4px solid #ffc107 !important;
            color: #ffecb3 !important;
        }
        
        /* Template Cards */
        .template-card{
            background: rgba(30,35,45,0.9) !important;
            border-color: #667eea !important;
        }
        .template-card:hover{
            background: rgba(102, 126, 234, 0.2) !important;
        }
        .template-card.selected{
            background: rgba(102, 126, 234, 0.3) !important;
        }
        
        /* Stats Cards */
        .stats-card{
            background: rgba(30,35,45,0.9) !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"]{
            background: rgba(10,15,20,0.92) !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
        }
        div[data-testid="stSidebarContent"]{ 
            background: transparent !important; 
        }
        
        /* Tabs */
        button[data-baseweb="tab"]{
            color: #e6e6e6 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"]{
            color: #667eea !important;
            border-bottom-color: #667eea !important;
        }
        
        /* Input Felder */
        input, textarea, select, .stSelectbox, .stTextInput, .stTextArea {
            background-color: rgba(30,35,45,0.9) !important;
            color: #e6e6e6 !important;
            border-color: rgba(255,255,255,0.1) !important;
        }
        
        /* Buttons */
        button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        
        button[kind="secondary"] {
            background: rgba(255,255,255,0.08) !important;
            color: #e6e6e6 !important;
            border-color: rgba(255,255,255,0.2) !important;
        }
        
        /* Expander */
        div[data-testid="stExpander"] {
            background-color: rgba(30,35,45,0.5) !important;
            border-color: rgba(255,255,255,0.1) !important;
        }
        
        /* Code Blocks */
        pre, code{
            background: rgba(17,24,39,0.9) !important;
            color: #e6e6e6 !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
        }
        
        /* Divider */
        hr {
            border-color: rgba(255,255,255,0.1) !important;
        }
        
        /* Info/Success/Error Boxes */
        div[data-testid="stAlert"] {
            background-color: rgba(30,35,45,0.8) !important;
            color: #e6e6e6 !important;
        }
        
        /* Download Button */
        div[data-testid="stDownloadButton"] button {
            background: rgba(102,126,234,0.2) !important;
            color: #e6e6e6 !important;
            border: 1px solid rgba(102,126,234,0.5) !important;
        }
        
        /* Json Viewer */
        .stJson {
            background-color: rgba(17,24,39,0.9) !important;
        }
        
        /* Spinner */
        div[data-testid="stSpinner"] > div {
            border-color: #667eea transparent transparent transparent !important;
        }
    }
</style>""", unsafe_allow_html=True)

# -----------------------------
# OpenAI Client
# -----------------------------
@st.cache_resource
def get_openai_client():
    if not OPENAI_API_KEY:
        st.error("⚠️ OPENAI_API_KEY nicht gefunden. Lege eine .env Datei an.")
        st.stop()
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"⚠️ OpenAI Paket fehlt: {e}")
        st.stop()

def clean_json_text(text: str) -> str:
    """Entfernt Markdown-Fences aus JSON-Antworten"""
    t = (text or "").strip()
    if t.startswith("```json"):
        t = t[len("```json"):].strip()
    if t.startswith("```"):
        t = t[len("```"):].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t

# -----------------------------
# Hauptgenerierungsfunktion
# -----------------------------
def generate_content_ideas(
    topic: str,
    tone: str,
    audience: str,
    content_type: str = "Alle Formate",
    keywords: str = "",
    length: str = "30-60 Sekunden",
    focus_areas: Optional[List[str]] = None,
    additional_context: str = "",
    healing_phase: str = "Alle Phasen",
    cta_type: str = "Engagement steigern",
    visual_style: str = "Flexibel",
    include_trigger_warning: bool = False,
    num_ideas: int = 3,
    template_name: Optional[str] = None,
) -> List[Dict]:
    """Generiert Content-Ideen mit erweiterten Parametern"""
    
    # Parameter-Hash für Caching
    params = {
        "topic": topic,
        "tone": tone,
        "audience": audience,
        "content_type": content_type,
        "keywords": keywords,
        "length": length,
        "focus_areas": focus_areas,
        "additional_context": additional_context,
        "healing_phase": healing_phase,
        "cta_type": cta_type,
        "visual_style": visual_style,
        "include_trigger_warning": include_trigger_warning,
        "num_ideas": num_ideas,
        "template_name": template_name
    }
    
    params_hash = create_parameters_hash(params)
    
    # Check Cache
    cached = check_cached_generation(params_hash)
    if cached:
        st.info("♻️ Geladene Ideen aus Cache!")
        return cached
    
    client = get_openai_client()
    focus_str = ", ".join(focus_areas) if focus_areas else "allgemeine Aufklärung"
    
    # Template-spezifische Anweisungen
    template_instructions = ""
    if template_name and template_name in CONTENT_TEMPLATES:
        template = CONTENT_TEMPLATES[template_name]
        template_instructions = f"""
        VERWENDE DIESES CONTENT-TEMPLATE: {template['name']}
        Struktur: {template['structure']}
        Hook-Formate: {', '.join(template['hook_formats'])}
        Beschreibung: {template['description']}
        """

    system_prompt = (
        "Du bist ein Experte für virale Social Media Content Creation spezialisiert auf Mental Health, "
        "Trauma-Aufklärung und Healing-Content. Du verstehst die Nuancen von narzisstischem Missbrauch "
        "und erstellst empowering, sensitiven Content, der gleichzeitig viral gehen kann. "
        "Du gibst Antworten IMMER als gültiges JSON-Array zurück."
    )

    trigger_warning_instruction = (
        "\n\nWICHTIG: Füge eine angemessene Trigger-Warnung am Anfang der Caption hinzu."
        if include_trigger_warning else ""
    )

    user_prompt = f"""
Erstelle {num_ideas} virale Instagram Reel-Ideen für einen Kanal über narzisstischen Missbrauch und Healing.

{template_instructions}

PARAMETER:
- Thema: {topic}
- Tonart: {tone}
- Zielgruppe: {audience}
- Healing-Phase: {healing_phase}
- Content-Typ: {content_type}
- Länge: {length}
- Visueller Stil: {visual_style}
- CTA-Fokus: {cta_type}
- Schwerpunkte: {focus_str}
- Keywords: {keywords}
- Zusätzlicher Kontext: {additional_context}{trigger_warning_instruction}

Für jede Idee benötige ich:
1. Einen ansprechenden Titel
2. 3 verschiedene Hook-Varianten (1-2 Sätze, aufmerksamkeitsstark)
3. Eine komplette Caption (mit Emojis, Formatierung, relevanten Hashtags)
4. Einen spezifischen CTA passend zum {cta_type}
5. Eine Viralitäts-Wahrscheinlichkeit (0-100)
6. Einen Engagement-Tipp
7. 8-12 relevante Hashtags
8. Vorschlag für visuellen Aufbau (Text Overlays, Szenen, etc.)
9. Empfohlene Posting-Zeit (basierend auf Zielgruppe)
10. Trending Audio Vorschlag (Genre/Mood)

WICHTIG: Sensitiv, unterstützend, empowering. Keine Stigmatisierung. Keine Diagnose-Versprechen.

Gib die Antwort als gültiges JSON Array in diesem Format zurück:
[
  {{
    "title": "Titel der Idee",
    "hooks": ["Hook 1", "Hook 2", "Hook 3"],
    "caption": "Komplette Caption mit Emojis und Formatierung...",
    "cta": "Spezifischer Call-to-Action",
    "viral_score": 85,
    "engagement_tip": "Diese Idee wird viral weil...",
    "hashtags": ["#hashtag1", "#hashtag2"],
    "visual_structure": "Beschreibung des visuellen Aufbaus",
    "best_posting_time": "z.B. Dienstag 19-21 Uhr",
    "audio_suggestion": "z.B. Emotional Piano / Trending Audio XY"
  }}
]
"""

    try:
        start_time = time.time()
        
        resp = generate_with_retry(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=OPENAI_MODEL,
            temperature=0.8,
            max_tokens=3500,
        )
        
        generation_time = time.time() - start_time

        content = resp.choices[0].message.content or ""
        content = clean_json_text(content)
        ideas = json.loads(content)

        if not isinstance(ideas, list):
            raise ValueError("Antwort ist kein JSON-Array.")

        # Fallbacks für fehlende Felder und Quality Check
        for idea in ideas:
            idea.setdefault("viral_score", random.randint(75, 95))
            idea.setdefault("engagement_tip", "Content performt gut durch starke Validation.")
            idea.setdefault("hooks", ["(Hook fehlt)"])
            idea.setdefault("hashtags", [])
            idea.setdefault("cta", "Teile deine Gedanken in den Kommentaren 💬")
            idea.setdefault("visual_structure", "Nicht spezifiziert")
            idea.setdefault("best_posting_time", "Abends 19-21 Uhr")
            idea.setdefault("audio_suggestion", "Emotionale Hintergrundmusik")
            
            # Quality Check
            idea = quality_check_idea(idea)

        # In Datenbank speichern
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO generations 
                (timestamp, parameters_hash, parameters, ideas, avg_viral_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    params_hash,
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(ideas, ensure_ascii=False),
                    sum(i.get("viral_score", 0) for i in ideas) / len(ideas)
                )
            )
            conn.commit()

        # Analytics tracken
        if "analytics_tracker" in st.session_state:
            st.session_state.analytics_tracker["total_generations"] += 1
            st.session_state.analytics_tracker["total_ideas"] += len(ideas)
            st.session_state.analytics_tracker["generation_times"].append(generation_time)
            
            # Topic Popularity
            st.session_state.analytics_tracker["topic_popularity"][topic] = \
                st.session_state.analytics_tracker["topic_popularity"].get(topic, 0) + 1
            
            # Tone Preferences
            st.session_state.analytics_tracker["tone_preferences"][tone] = \
                st.session_state.analytics_tracker["tone_preferences"].get(tone, 0) + 1

        return ideas

    except json.JSONDecodeError as e:
        st.error(f"❌ JSON-Fehler: {e}")
        if 'content' in locals():
            with st.expander("📋 Rohe Antwort anzeigen"):
                st.code(content, language="text")
        return []
    except Exception as e:
        st.error(f"❌ Fehler: {e}")
        return []

# -----------------------------
# History Management
# -----------------------------
def save_to_history(ideas: List[Dict], params: Dict):
    """Speichert generierte Ideen in Session State"""
    entry = {
        "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "params": params,
        "ideas": ideas,
        "avg_viral_score": sum(i.get("viral_score", 0) for i in ideas) / len(ideas) if ideas else 0,
        "quality_check": batch_quality_check(ideas)
    }
    st.session_state.generated_history.insert(0, entry)
    # Limit auf letzte 10
    st.session_state.generated_history = st.session_state.generated_history[:10]

def load_history_from_db(limit: int = 10) -> List[Dict]:
    """Lädt History aus der Datenbank"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM generations ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "id": row['id'],
                    "timestamp": datetime.fromisoformat(row['timestamp']).strftime("%d.%m.%Y %H:%M"),
                    "parameters": json.loads(row['parameters']),
                    "ideas": json.loads(row['ideas']),
                    "avg_viral_score": row['avg_viral_score'],
                    "user_rating": row['user_rating'],
                    "feedback": row['feedback']
                })
            return history
    except Exception as e:
        st.warning(f"⚠️ Datenbank-Fehler: {e}")
        return []

# -----------------------------
# Export Funktionen
# -----------------------------
def export_ideas_to_json(ideas: List[Dict]) -> str:
    """Exportiert Ideen als JSON-String"""
    return json.dumps(ideas, ensure_ascii=False, indent=2)

def export_ideas_to_csv(ideas: List[Dict]) -> str:
    """Exportiert Ideen als CSV-String"""
    data = []
    for idea in ideas:
        data.append({
            'Title': idea.get('title', ''),
            'Hooks': ' | '.join(idea.get('hooks', [])),
            'Caption_Preview': (idea.get('caption', '')[:200] + '...') if len(idea.get('caption', '')) > 200 else idea.get('caption', ''),
            'CTA': idea.get('cta', ''),
            'Viral_Score': idea.get('viral_score', 0),
            'Quality_Score': idea.get('quality_score', 0),
            'Quality_Status': idea.get('quality_status', ''),
            'Hashtags': ' '.join(idea.get('hashtags', [])),
            'Posting_Time': idea.get('best_posting_time', ''),
            'Audio': idea.get('audio_suggestion', ''),
            'Visual_Structure': idea.get('visual_structure', ''),
            'Engagement_Tip': idea.get('engagement_tip', '')
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False, encoding='utf-8-sig')

def export_content_calendar(ideas: List[Dict], days: int = 7) -> str:
    """Exportiert einen Content-Kalender"""
    calendar = []
    start_date = datetime.now()
    
    for i in range(min(days, len(ideas))):
        idea = ideas[i]
        post_date = (start_date + timedelta(days=i)).strftime("%d.%m.%Y")
        
        calendar.append({
            'Date': post_date,
            'Day': (start_date + timedelta(days=i)).strftime("%A"),
            'Time': idea.get('best_posting_time', '19:00'),
            'Title': idea.get('title', ''),
            'Theme': 'Educational' if i % 3 == 0 else 'Storytelling' if i % 3 == 1 else 'Quick Tip',
            'Viral_Score': idea.get('viral_score', 0),
            'Primary_Hashtags': ' '.join(idea.get('hashtags', [])[:5]),
            'Audio_Suggestion': idea.get('audio_suggestion', ''),
            'Visual_Style': idea.get('visual_structure', ''),
            'CTA_Type': idea.get('cta', '')[0:50]
        })
    
    df = pd.DataFrame(calendar)
    return df.to_csv(index=False, encoding='utf-8-sig')

# -----------------------------
# A/B Test Funktionen
# -----------------------------
def generate_ab_test_variations(idea: Dict, num_variations: int = 3) -> List[Dict]:
    """Generiert verschiedene Versionen für A/B Testing"""
    variations = []
    base_hooks = idea.get('hooks', [])
    
    # Verschiedene CTA-Optionen
    cta_variations = [
        "Teile deine Erfahrung in den Kommentaren 👇",
        "Speicher dir das für später ⬇️",
        "Folge für mehr Insights ➡️",
        "Kommentiere 'JA' wenn du das kennst 💬",
        "Tag jemanden, dem das helfen könnte 👥",
        "Welcher Punkt spricht dich am meisten an? 📝"
    ]
    
    # Verschiedene Hook-Strategien
    hook_strategies = [
        lambda hooks: random.sample(hooks, min(2, len(hooks))),  # Zwei beste Hooks
        lambda hooks: [random.choice(hooks) if hooks else ""],   # Ein Hook
        lambda hooks: [hooks[0] + " (Teil 1)", hooks[1] + " (Teil 2)"] if len(hooks) >= 2 else hooks  # Serie
    ]
    
    for i in range(num_variations):
        variation = idea.copy()
        variation['title'] = f"{idea.get('title', '')} - Variante {i+1}"
        
        # Unterschiedliche Hooks
        if base_hooks:
            strategy = random.choice(hook_strategies)
            variation['hooks'] = strategy(base_hooks)
        
        # Unterschiedliche CTAs
        variation['cta'] = random.choice(cta_variations)
        
        # Leicht veränderte Hashtags
        hashtags = idea.get('hashtags', [])
        if len(hashtags) > 5:
            variation['hashtags'] = random.sample(hashtags, len(hashtags) - 2)
        
        # Leicht veränderter Viral Score
        original_score = idea.get('viral_score', 80)
        variation['viral_score'] = max(0, min(100, original_score + random.randint(-5, 5)))
        
        variations.append(variation)
    
    return variations

# -----------------------------
# Batch Processing
# -----------------------------
def generate_content_batch(topics: List[str], base_params: Dict) -> List[Dict]:
    """Generiert Content für mehrere Themen auf einmal"""
    all_ideas = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, topic in enumerate(topics):
        status_text.text(f"Generiere Ideen für: {topic}")
        
        params = base_params.copy()
        params["topic"] = topic
        params["num_ideas"] = 1  # Nur eine Idee pro Thema für Batch
        
        ideas = generate_content_ideas(**params)
        all_ideas.extend(ideas)
        
        progress_bar.progress((i + 1) / len(topics))
    
    status_text.text("✅ Batch-Generierung abgeschlossen!")
    return all_ideas

# -----------------------------
# UI Hauptfunktionen
# -----------------------------
def display_analytics_dashboard():
    """Zeigt Analytics Dashboard an"""
    st.header("📈 Analytics Dashboard")
    
    if "analytics_tracker" not in st.session_state:
        st.info("Noch keine Analytics-Daten verfügbar.")
        return
    
    analytics = st.session_state.analytics_tracker
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Generierungen", analytics.get("total_generations", 0))
    with col2:
        st.metric("Total Ideen", analytics.get("total_ideas", 0))
    with col3:
        avg_time = sum(analytics.get("generation_times", [0])) / max(len(analytics.get("generation_times", [1])), 1)
        st.metric("Ø Generierungszeit", f"{avg_time:.1f}s")
    with col4:
        st.metric("Ø Viral Score", f"{analytics.get('avg_viral_score', 0):.0f}/100")
    
    # Topic Popularity Chart
    st.subheader("📊 Beliebte Themen")
    topic_data = analytics.get("topic_popularity", {})
    if topic_data:
        topics_df = pd.DataFrame(list(topic_data.items()), columns=['Thema', 'Anzahl'])
        st.bar_chart(topics_df.set_index('Thema'))
    
    # Tone Preferences
    st.subheader("🎭 Bevorzugte Tonarten")
    tone_data = analytics.get("tone_preferences", {})
    if tone_data:
        col1, col2 = st.columns(2)
        with col1:
            for tone, count in sorted(tone_data.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"**{tone}:** {count}")
        
        with col2:
            tones_df = pd.DataFrame(list(tone_data.items()), columns=['Tonart', 'Anzahl'])
            st.dataframe(tones_df.sort_values('Anzahl', ascending=False).head(10), use_container_width=True)

# -----------------------------
# Haupt-UI
# -----------------------------
def main():
    # Datenbank initialisieren
    init_database()
    
    st.markdown('<h1 class="main-title">🎯 Instagram Reel Generator PRO</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Virale Content-Ideen für Narzissmus-Aufklärung und Healing</p>', unsafe_allow_html=True)
    
    # Tabs für bessere Organisation
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Generator", "📊 History", "📈 Analytics", "🔄 Batch", "ℹ️ Info"])
    
    with tab1:
        with st.sidebar:
            st.header("🎨 Content-Einstellungen")
            
            # Template Auswahl
            with st.expander("📋 Content Templates", expanded=True):
                selected_template = st.selectbox(
                    "Wähle ein Template",
                    ["Kein Template", "educational", "storytelling", "quick_tips", "myth_busting"],
                    format_func=lambda x: CONTENT_TEMPLATES.get(x, {}).get("name", "Kein Template") if x != "Kein Template" else "Kein Template"
                )
                
                if selected_template != "Kein Template":
                    template = CONTENT_TEMPLATES.get(selected_template, {})
                    st.info(f"""
                    **{template.get('name', '')}**
                    {template.get('description', '')}
                    
                    **Struktur:** {template.get('structure', '')}
                    **Länge:** {template.get('length', '')}
                    **Hook-Formate:** {', '.join(template.get('hook_formats', []))}
                    """)
            
            # Basis-Einstellungen
            with st.expander("📌 Basis-Einstellungen", expanded=True):
                topic = st.selectbox(
                    "Hauptthema *",
                    ["", "Healing & Recovery", "Red Flags erkennen", "Gaslighting", 
                     "Gesunde Grenzen", "Selbstliebe", "Aufklärung & Info", 
                     "Motivation & Hoffnung", "Validation", "Co-Abhängigkeit",
                     "Trauma Bonding", "No Contact", "Emotionale Erpressung",
                     "Narzisstische Eltern", "Selbstwert aufbauen", "Wut & Trauer"],
                )

                tone = st.selectbox(
                    "Tonart *",
                    ["", "Mut machend", "Bildend", "Empathisch", "Bestimmt", 
                     "Hoffnungsvoll", "Validierend", "Direkt", "Poetisch", "Ermächtigend"],
                )

                audience = st.selectbox(
                    "Zielgruppe *",
                    ["", "Überlebende", "Im Healing-Prozess", "Interessierte", 
                     "Angehörige & Freunde", "Therapeuten & Coaches", "Neu Betroffene"],
                )
                
                healing_phase = st.selectbox(
                    "Healing-Phase",
                    ["Alle Phasen", "Frisch aus Beziehung", "Im Prozess", 
                     "Fortgeschritten", "Helfer/Supporter", "Rückfall-Phase"],
                )

            # Erweiterte Einstellungen
            with st.expander("⚙️ Erweiterte Einstellungen"):
                content_type = st.selectbox(
                    "Content-Typ",
                    ["Alle Formate", "Bildend (3-2-1 Format)", "Storytelling", 
                     "Quick Tips", "Myth Busting", "Validation", "Q&A", "Checkliste"],
                )
                
                visual_style = st.selectbox(
                    "Visueller Stil",
                    ["Flexibel", "Talking Head", "Text Overlay Heavy", 
                     "B-Roll Focus", "Animation", "Hybrid", "Stock Footage"],
                )
                
                cta_type = st.selectbox(
                    "CTA-Fokus",
                    ["Engagement steigern", "Kommentieren", "Teilen", 
                     "Speichern", "Link in Bio", "Community aufbauen", "DM für Hilfe"],
                )

                length = st.selectbox(
                    "Geschätzte Länge", 
                    ["15-30 Sekunden", "30-60 Sekunden", "60-90 Sekunden", "90+ Sekunden"],
                    index=1
                )
                
                num_ideas = st.slider("Anzahl Ideen", 1, 5, 3)
                
                include_trigger_warning = st.checkbox(
                    "⚠️ Trigger-Warnung einbeziehen",
                    help="Fügt automatisch eine Trigger-Warnung hinzu",
                    value=True
                )
                
                enable_quality_check = st.checkbox(
                    "✅ Automatische Qualitätsprüfung",
                    help="Prüft generierte Ideen auf Qualitätsmerkmale",
                    value=True
                )

            # Content-Details
            with st.expander("🔍 Content-Details"):
                keywords = st.text_input(
                    "Schlüsselwörter",
                    placeholder="z.B. Love Bombing, Hoovering, Flying Monkeys, Grey Rock",
                )

                focus_areas = st.multiselect(
                    "Content-Schwerpunkte (max. 3)",
                    ["Bewusstsein schaffen", "Validierung", "Handlungsschritte", 
                     "Emotionale Unterstützung", "Bildung", "Heilungstipps",
                     "Boundary-Setting", "Self-Care", "Warnsignale", "Community"],
                    max_selections=3,
                )

                additional_context = st.text_area(
                    "Zusätzlicher Kontext",
                    placeholder="Spezifische Situation, aktuelle Trends, persönliche Note...",
                    height=100,
                )
                
                # A/B Testing Option
                enable_ab_testing = st.checkbox(
                    "🔬 A/B Testing Varianten generieren",
                    help="Erstellt 3 leicht unterschiedliche Versionen jeder Idee"
                )

        # Generate Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_button = st.button(
                "🚀 Content-Ideen generieren",
                type="primary",
                use_container_width=True,
                disabled=not all([topic, tone, audience]),
            )

        if generate_button:
            if not all([topic, tone, audience]):
                st.error("❌ Bitte fülle alle Pflichtfelder aus!")
            else:
                with st.spinner("🎨 Generiere virale Content-Ideen..."):
                    ideas = generate_content_ideas(
                        topic=topic,
                        tone=tone,
                        audience=audience,
                        content_type=content_type,
                        keywords=keywords,
                        length=length,
                        focus_areas=focus_areas,
                        additional_context=additional_context,
                        healing_phase=healing_phase,
                        cta_type=cta_type,
                        visual_style=visual_style,
                        include_trigger_warning=include_trigger_warning,
                        num_ideas=num_ideas,
                        template_name=selected_template if selected_template != "Kein Template" else None,
                    )

                if ideas:
                    # Quality Check durchführen
                    if enable_quality_check:
                        quality_report = batch_quality_check(ideas)
                        ideas = quality_report['checked_ideas']
                    
                    # A/B Testing Varianten generieren
                    ab_variations = []
                    if enable_ab_testing:
                        for idea in ideas:
                            variations = generate_ab_test_variations(idea, num_variations=3)
                            ab_variations.extend(variations)
                    
                    # Speichere in History
                    params = {
                        "topic": topic, "tone": tone, "audience": audience,
                        "healing_phase": healing_phase, "content_type": content_type,
                        "template": selected_template
                    }
                    save_to_history(ideas, params)
                    
                    # Metriken anzeigen
                    col1, col2, col3, col4 = st.columns(4)
                    avg_score = sum(i.get("viral_score", 0) for i in ideas) / len(ideas)
                    avg_quality = sum(i.get("quality_score", 0) for i in ideas) / len(ideas) if enable_quality_check else 0
                    
                    with col1:
                        st.markdown(f"""<div class="metric-card">
                            <h3>{len(ideas)}</h3><p>Ideen generiert</p></div>""", 
                            unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""<div class="metric-card">
                            <h3>{avg_score:.0f}/100</h3><p>Ø Viral-Score</p></div>""", 
                            unsafe_allow_html=True)
                    with col3:
                        if enable_quality_check:
                            st.markdown(f"""<div class="metric-card">
                                <h3>{avg_quality:.0f}/100</h3><p>Ø Qualität</p></div>""", 
                                unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""<div class="metric-card">
                            <h3>{sum(len(i.get('hashtags',[])) for i in ideas)}</h3>
                            <p>Hashtags total</p></div>""", 
                            unsafe_allow_html=True)

                    st.success(f"✅ {len(ideas)} Content-Ideen erfolgreich generiert!")
                    
                    # Quality Report anzeigen
                    if enable_quality_check and quality_report['common_issues']:
                        with st.expander("📊 Qualitätsreport", expanded=True):
                            col_q1, col_q2 = st.columns(2)
                            with col_q1:
                                st.metric("Durchschnittliche Qualität", f"{quality_report['avg_quality_score']}%")
                                st.metric("Exzellente Ideen", quality_report['excellent_count'])
                            with col_q2:
                                st.metric("Verbesserungsbedarf", quality_report['improvement_count'])
                                if quality_report['common_issues']:
                                    st.write("**Häufige Probleme:**")
                                    for issue, count in quality_report['common_issues'][:3]:
                                        st.write(f"- {issue}: {count}x")
                    
                    # Export-Buttons
                    with st.expander("💾 Export-Optionen", expanded=True):
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        
                        with col_exp1:
                            json_export = export_ideas_to_json(ideas)
                            st.download_button(
                                "📥 Als JSON",
                                json_export,
                                file_name=f"reel_ideas_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col_exp2:
                            csv_export = export_ideas_to_csv(ideas)
                            st.download_button(
                                "📊 Als CSV",
                                csv_export,
                                file_name=f"reel_ideas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col_exp3:
                            calendar_export = export_content_calendar(ideas, days=7)
                            st.download_button(
                                "📅 Content-Kalender",
                                calendar_export,
                                file_name=f"content_calendar_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    # A/B Testing Varianten anzeigen
                    if enable_ab_testing and ab_variations:
                        st.subheader("🔬 A/B Testing Varianten")
                        for i, variation in enumerate(ab_variations, 1):
                            with st.expander(f"Variante {i}: {variation.get('title', '')}"):
                                st.write(f"**Hook:** {variation.get('hooks', [''])[0]}")
                                st.write(f"**CTA:** {variation.get('cta', '')}")
                                st.write(f"**Viral Score:** {variation.get('viral_score', 0)}/100")
                    
                    # Ideen anzeigen
                    for idx, idea in enumerate(ideas, 1):
                        score = int(idea.get("viral_score", 80))
                        quality = idea.get("quality_score", 0)
                        quality_status = idea.get("quality_status", "Unknown")
                        
                        # Badge für Viral Score
                        if score >= 90:
                            badge_class, fire_emoji = "viral-badge-high", "🔥"
                        elif score >= 80:
                            badge_class, fire_emoji = "viral-badge-medium", "⚡"
                        else:
                            badge_class, fire_emoji = "viral-badge-low", "✅"
                        
                        # Badge für Quality Score
                        if quality_status == "Excellent":
                            quality_badge = '<span class="quality-badge-excellent">Excellent</span>'
                        elif quality_status == "Good":
                            quality_badge = '<span class="quality-badge-good">Good</span>'
                        else:
                            quality_badge = '<span class="quality-badge-improve">Improve</span>'
                        
                        st.markdown(f"""
                            <div class="content-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <h3>#{idx}: {idea.get('title','(ohne Titel)')}</h3>
                                    <div style="display:flex; gap:10px; align-items:center;">
                                        {quality_badge}
                                        <span style="{badge_class.replace('viral-badge-', 'background:')} color:white;padding:8px 16px;border-radius:20px;font-weight:700;">
                                        {fire_emoji} {score}/100</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Quality Issues anzeigen
                        if idea.get('quality_issues'):
                            with st.expander("⚠️ Qualitätshinweise", expanded=False):
                                st.warning(f"**Qualität:** {idea.get('quality_score', 0)}%")
                                for issue in idea.get('quality_issues', []):
                                    st.write(f"- {issue}")

                        # Hooks
                        with st.expander("🎣 Hook-Varianten", expanded=True):
                            for h_idx, hook in enumerate(idea.get("hooks", []), 1):
                                col_hook, col_copy = st.columns([5, 1])
                                with col_hook:
                                    st.info(f"**Hook {h_idx}:** {hook}")
                                with col_copy:
                                    if st.button("📋", key=f"copy_hook_{idx}_{h_idx}"):
                                        st.write("Kopiert!")
                                        # In Streamlit Cloud müsste man Clipboard API verwenden

                        # Caption + CTA
                        with st.expander("📝 Caption & CTA"):
                            caption_text = idea.get('caption', '')
                            st.text_area(
                                "Caption:",
                                caption_text,
                                height=200,
                                key=f"caption_{idx}"
                            )
                            st.success(f"**CTA:** {idea.get('cta', '')}")

                        # Neue Features
                        col1, col2 = st.columns(2)
                        with col1:
                            with st.expander("🎬 Visueller Aufbau"):
                                st.write(idea.get('visual_structure', 'Nicht angegeben'))
                        with col2:
                            with st.expander("⏰ Posting-Empfehlung"):
                                st.write(f"**Beste Zeit:** {idea.get('best_posting_time', 'N/A')}")
                                st.write(f"**Audio:** {idea.get('audio_suggestion', 'N/A')}")

                        # Engagement
                        st.info(f"💡 **Engagement-Tipp:** {idea.get('engagement_tip', '')}")

                        # Hashtags
                        with st.expander("🏷️ Hashtags"):
                            hashtags = idea.get('hashtags', [])
                            st.code(' '.join(hashtags), language='text')
                        
                        # User Feedback
                        with st.expander("⭐ Bewertung", expanded=False):
                            col_fb1, col_fb2, col_fb3 = st.columns(3)
                            with col_fb1:
                                if st.button("👍 Gut", key=f"good_{idx}"):
                                    st.success("Danke für dein Feedback!")
                            with col_fb2:
                                if st.button("👎 Nicht hilfreich", key=f"bad_{idx}"):
                                    st.info("Wir arbeiten an Verbesserungen!")
                            with col_fb3:
                                if st.button("🔄 Ähnliche generieren", key=f"similar_{idx}"):
                                    st.info("Diese Funktion ist in Entwicklung!")
                        
                        st.divider()

    with tab2:
        st.header("📊 Generierungs-History")
        
        # History aus Datenbank laden
        db_history = load_history_from_db(limit=20)
        
        if not db_history and not st.session_state.generated_history:
            st.info("Noch keine Ideen generiert. Starte im Generator-Tab!")
        else:
            # Aktuelle Session History
            if st.session_state.generated_history:
                st.subheader("🗓️ Aktuelle Session")
                for entry in st.session_state.generated_history:
                    display_history_entry(entry)
            
            # Datenbank History
            if db_history:
                st.subheader("💾 Gespeicherte History")
                for entry in db_history:
                    display_db_history_entry(entry)
    
    with tab3:
        display_analytics_dashboard()
    
    with tab4:
        st.header("🔄 Batch-Generierung")
        
        with st.expander("⚙️ Batch-Einstellungen", expanded=True):
            batch_topics = st.multiselect(
                "Themen für Batch-Generierung",
                ["Healing & Recovery", "Red Flags erkennen", "Gaslighting", 
                 "Gesunde Grenzen", "Selbstliebe", "Aufklärung & Info",
                 "Motivation & Hoffnung", "Validation", "Co-Abhängigkeit"],
                default=["Healing & Recovery", "Red Flags erkennen"]
            )
            
            batch_tone = st.selectbox(
                "Tonart für alle Themen",
                ["Mut machend", "Bildend", "Empathisch", "Hoffnungsvoll"],
                key="batch_tone"
            )
            
            batch_audience = st.selectbox(
                "Zielgruppe für alle Themen",
                ["Überlebende", "Im Healing-Prozess", "Interessierte"],
                key="batch_audience"
            )
        
        if st.button("🚀 Batch generieren", type="primary", use_container_width=True):
            if not batch_topics:
                st.error("❌ Bitte wähle mindestens ein Thema!")
            else:
                base_params = {
                    "tone": batch_tone,
                    "audience": batch_audience,
                    "content_type": "Alle Formate",
                    "length": "30-60 Sekunden",
                    "healing_phase": "Alle Phasen",
                    "cta_type": "Engagement steigern",
                    "include_trigger_warning": True
                }
                
                batch_ideas = generate_content_batch(batch_topics, base_params)
                
                if batch_ideas:
                    st.success(f"✅ {len(batch_ideas)} Ideen für {len(batch_topics)} Themen generiert!")
                    
                    # Batch-Export
                    with st.expander("💾 Batch exportieren"):
                        batch_json = export_ideas_to_json(batch_ideas)
                        batch_csv = export_ideas_to_csv(batch_ideas)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Batch als JSON",
                                batch_json,
                                file_name=f"batch_ideas_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        with col2:
                            st.download_button(
                                "📊 Batch als CSV",
                                batch_csv,
                                file_name=f"batch_ideas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    # Batch-Übersicht
                    st.subheader("📋 Batch-Übersicht")
                    for topic in batch_topics:
                        topic_ideas = [idea for idea in batch_ideas if topic.lower() in idea.get('title', '').lower() or 
                                      topic.lower() in idea.get('caption', '').lower()]
                        with st.expander(f"{topic} ({len(topic_ideas)} Ideen)"):
                            for idea in topic_ideas:
                                st.write(f"• {idea.get('title', '')} - Score: {idea.get('viral_score', 0)}/100")
    
    with tab5:
        st.header("ℹ️ Tool-Information")
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            ### 🎯 Features
            - **Erweiterte Targeting-Optionen** für verschiedene Healing-Phasen
            - **Visueller Stil-Auswahl** für unterschiedliche Content-Formate
            - **CTA-Optimierung** für spezifische Engagement-Ziele
            - **Trigger-Warnungen** für sensible Inhalte
            - **Export-Funktion** zum Speichern deiner Ideen
            - **History-Tracking** der letzten 10 Generierungen
            - **Quality Control** automatische Qualitätsprüfung
            - **A/B Testing** verschiedene Versionen generieren
            - **Batch Processing** mehrere Themen auf einmal
            - **Analytics Dashboard** Performance-Tracking
            
            ### 🔒 Sicherheit & Ethik
            - Keine Diagnosen stellen
            - Trigger-Warnungen nutzen
            - Professionelle Hilfe empfehlen
            - Respektvoll mit Überlebenden umgehen
            - Datenschutz beachten
            """)
        
        with col_info2:
            st.markdown("""
            ### 💡 Best Practices
            1. **Authentizität**: Bleibe bei deiner persönlichen Story
            2. **Konsistenz**: Poste regelmäßig (3-5x/Woche)
            3. **Engagement**: Antworte auf Kommentare binnen 24h
            4. **Hashtags**: Mix aus großen (100k+) und Nischen-Tags (1-10k)
            5. **Posting-Zeit**: Teste verschiedene Zeiten für deine Audience
            6. **Quality over Quantity**: Lieber weniger, aber hochwertige Posts
            7. **Community Building**: Baue eine unterstützende Community auf
            8. **Tracking**: Verfolge, welche Inhalte gut performen
            
            ### 🚀 Pro-Tipps
            - Nutze die A/B Testing Funktion für Optimierung
            - Exportiere deinen Content-Kalender für Planung
            - Überprüfe die Quality Scores für bessere Ergebnisse
            - Nutze Batch Processing für Content-Wochen
            - Tracke deine Analytics für Insights
            """)
        
        # System-Info
        with st.expander("⚙️ System-Information"):
            st.write(f"**OpenAI Model:** {OPENAI_MODEL}")
            st.write(f"**Datenbank:** reel_generator.db")
            st.write(f"**Session Generierungen:** {len(st.session_state.generated_history)}")
            if "analytics_tracker" in st.session_state:
                st.write(f"**Total Ideen:** {st.session_state.analytics_tracker.get('total_ideas', 0)}")
            st.write(f"**Letztes Update:** {datetime.now().strftime('%d.%m.%Y %H:%M')}")

def display_history_entry(entry: Dict):
    """Zeigt einen History-Eintrag an"""
    with st.expander(
        f"📅 {entry['timestamp']} - {entry['params']['topic']} "
        f"(Ø Score: {entry['avg_viral_score']:.0f}/100)"
    ):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Parameter:**")
            st.json(entry['params'])
        with col2:
            st.write(f"**{len(entry['ideas'])} Ideen generiert**")
            if 'quality_check' in entry:
                st.write(f"**Qualität:** {entry['quality_check']['avg_quality_score']}%")
            
            # Export dieser spezifischen Ideen
            if st.button("📥 Exportieren", key=f"export_{entry['timestamp']}"):
                json_data = export_ideas_to_json(entry['ideas'])
                st.download_button(
                    "Download",
                    json_data,
                    file_name=f"history_{entry['timestamp'].replace(':', '-')}.json",
                    mime="application/json"
                )
        
        for i, idea in enumerate(entry['ideas'], 1):
            st.write(f"**{i}.** {idea.get('title', 'N/A')} - "
                    f"Score: {idea.get('viral_score', 0)}/100")
            if idea.get('quality_score'):
                st.write(f"   Qualität: {idea.get('quality_score')}% - {idea.get('quality_status')}")

def display_db_history_entry(entry: Dict):
    """Zeigt einen Datenbank-History-Eintrag an"""
    with st.expander(
        f"💾 {entry['timestamp']} - {entry['parameters'].get('topic', 'Unknown')} "
        f"(Score: {entry['avg_viral_score']:.0f}/100)"
    ):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Parameter:**")
            st.json(entry['parameters'])
        with col2:
            ideas = entry['ideas']
            st.write(f"**{len(ideas)} Ideen**")
            
            # User Rating anzeigen
            if entry['user_rating'] > 0:
                st.write(f"**Bewertung:** {'⭐' * entry['user_rating']}")
            
            if entry['feedback']:
                st.write(f"**Feedback:** {entry['feedback']}")
            
            # Export Button
            json_data = export_ideas_to_json(ideas)
            st.download_button(
                "📥 Exportieren",
                json_data,
                file_name=f"db_history_{entry['id']}.json",
                mime="application/json",
                key=f"db_export_{entry['id']}"
            )
        
        # Ideen anzeigen
        for i, idea in enumerate(ideas[:3], 1):  # Nur erste 3 zeigen
            st.write(f"**{i}.** {idea.get('title', 'N/A')} - Score: {idea.get('viral_score', 0)}/100")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7f8c8d; padding:20px;">
  <p><strong>Instagram Reel Generator PRO für Narzissmus-Aufklärung</strong></p>
  <p>Empowering Content für Überlebende und Aufklärung 💙</p>
  <p style="font-size:12px;">Powered by OpenAI | Erstellt mit Streamlit | v2.0</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()