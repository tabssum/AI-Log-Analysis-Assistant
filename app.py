import hashlib
import html
import json
from pathlib import Path
import re
import pandas as pd
import streamlit as st

from chunker import create_event_chunks
from embeddings import GeminiEmbeddingError
from llm import (
    DEFAULT_GEMINI_MODEL,
    GeminiError,
    RetrievedEvidence,
    generate_log_analysis,
)
from log_parser import ParsedLogEntry, parse_log_lines
from vector_store import ChromaVectorStore

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
is_parsed_view = st.query_params.get("view") == "parsed"
st.set_page_config(
    page_title="AI Log Analysis Assistant",
    page_icon="🛡️",
    layout="wide" if is_parsed_view else "centered",
    initial_sidebar_state="collapsed",
)

# Shared cache setup
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "latest_parsed_log.json"
_SHARED_CACHE = {}

# -----------------------------------------------------------------------------
# Enterprise-Grade Professional CSS Theme
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .block-container {
        max-width: 1040px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }

    /* Professional Hero Banner */
    .pro-header {
        background: linear-gradient(180deg, #0f172a 0%, #090d16 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 26px 30px;
        margin-bottom: 22px;
        color: #f8fafc;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .pro-header-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .pro-title-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .pro-title-icon {
        font-size: 24px;
        line-height: 1;
    }
    .pro-title-text {
        font-size: 23px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .pro-badge-ready {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 9999px;
        padding: 3px 11px;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-dot {
        width: 6.5px;
        height: 6.5px;
        background: #34d399;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #34d399;
    }
    .pro-subtitle {
        color: #94a3b8;
        font-size: 14.5px;
        line-height: 1.55;
        margin-bottom: 16px;
        max-width: 760px;
    }
    .pro-tags-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .pro-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 7px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 500;
        color: #cbd5e1;
        transition: all 0.2s ease;
    }
    .pro-tag:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Metrics Summary Bar */
    .metrics-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 18px 0 20px 0;
    }
    @media (max-width: 680px) {
        .metrics-bar {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .metric-cell {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.14);
        border-radius: 10px;
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
    }
    .metric-val {
        font-size: 16px;
        font-weight: 600;
        color: #e2e8f0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .metric-val-danger {
        color: #f87171;
    }
    .metric-val-warn {
        color: #fbbf24;
    }
    .metric-val-success {
        color: #34d399;
    }

    /* High-Visibility Hero Banner for Parsed Log Fields */
    .parsed-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.16) 0%, rgba(99, 102, 241, 0.18) 100%);
        border: 2px solid #3b82f6;
        border-radius: 14px;
        padding: 18px 24px;
        margin: 18px 0 22px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        box-shadow: 0 8px 30px -4px rgba(37, 99, 235, 0.35);
    }
    @media (max-width: 680px) {
        .parsed-banner {
            flex-direction: column;
            align-items: flex-start;
        }
    }
    .parsed-banner-info {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    .parsed-banner-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.01em;
    }
    .parsed-banner-tag {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 9999px;
        padding: 2px 9px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .parsed-banner-desc {
        font-size: 13.5px;
        color: #cbd5e1;
        line-height: 1.45;
    }
    .parsed-hero-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        border: 1.5px solid rgba(255, 255, 255, 0.35) !important;
        box-shadow: 0 6px 22px rgba(37, 99, 235, 0.5) !important;
        text-decoration: none !important;
        white-space: nowrap !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer;
        flex-shrink: 0;
    }
    .parsed-hero-btn:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%) !important;
        box-shadow: 0 8px 28px rgba(37, 99, 235, 0.75) !important;
        transform: translateY(-2px) scale(1.02) !important;
        color: #ffffff !important;
    }
    .ext-badge {
        background: rgba(255, 255, 255, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 12px;
        font-weight: 600;
        color: #ffffff;
    }

    /* Terminal Console Frame */
    .console-frame {
        border: 1px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
        background: #090d16;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
        margin-top: 14px;
    }
    .console-topbar {
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .console-dots {
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .dot {
        width: 10.5px;
        height: 10.5px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-red { background: #ef4444; opacity: 0.85; }
    .dot-yellow { background: #f59e0b; opacity: 0.85; }
    .dot-green { background: #10b981; opacity: 0.85; }
    .console-meta {
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 12px;
        color: #94a3b8;
    }

    /* Console Content & Gutter */
    .console-body {
        font-family: 'JetBrains Mono', Consolas, 'SFMono-Regular', Menlo, monospace;
        font-size: 12.5px;
        line-height: 1.65;
        color: #e2e8f0;
        padding: 14px 18px;
        max-height: 480px;
        overflow-y: auto;
        overflow-x: auto;
        white-space: pre;
    }
    .log-row {
        display: flex;
        gap: 14px;
        min-height: 20px;
    }
    .log-row:hover {
        background: rgba(255, 255, 255, 0.03);
    }
    .ln {
        color: #475569;
        user-select: none;
        width: 42px;
        text-align: right;
        flex-shrink: 0;
        font-size: 11.5px;
    }
    .msg {
        white-space: pre-wrap;
        word-break: break-all;
    }

    /* Token Highlighting */
    .tok-err { color: #f87171; font-weight: 600; }
    .tok-warn { color: #fbbf24; font-weight: 600; }
    .tok-info { color: #38bdf8; }
    .tok-ts { color: #64748b; }

    /* Empty State Box */
    .empty-upload-box {
        text-align: center;
        padding: 44px 20px;
        border: 2px dashed rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        margin-top: 18px;
        background: rgba(128, 128, 128, 0.02);
    }
    .empty-upload-title {
        font-size: 16px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 6px;
    }
    .empty-upload-sub {
        font-size: 13.5px;
        color: #64748b;
        max-width: 480px;
        margin: 0 auto;
    }

    /* AI Analysis Response Card */
    .ai-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 22px 24px;
        margin: 18px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .ai-card-title {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #38bdf8;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* Custom Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 14px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: rgba(56, 189, 248, 0.05);
    }

    /* Primary Form Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 9px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
        padding: 8px 20px !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Keep native Streamlit controls consistent with the dark application shell. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #090d16;
        color: #e2e8f0;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background-color: #0f172a;
        color: #e2e8f0;
        border-color: #334155;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stButton"] button {
        background-color: #1e293b;
        color: #e2e8f0;
        border-color: #475569;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stButton"] button:hover {
        background-color: #334155;
        color: #f8fafc;
        border-color: #64748b;
    }
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {
        background-color: #0f172a;
        border: 1px solid #334155;
    }
    [data-baseweb="popover"] [role="option"] {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background-color: #1e293b;
        color: #f8fafc;
    }
    [data-testid="stAlert"],
    [data-testid="stStatusWidget"],
    [data-testid="stExpander"] details,
    [data-testid="stCode"] {
        background-color: #0f172a;
        border-color: #334155;
        color: #e2e8f0;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label {
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background-color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_file_size(size_bytes: int) -> str:
    """Formats file size into standard human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def highlight_log_line(raw_line: str) -> str:
    """Safely escapes HTML and applies token highlighting for severity and timestamps."""
    escaped = html.escape(raw_line)
    escaped = re.sub(r"\b(CRITICAL|FATAL|ERROR)\b", r"<span class='tok-err'>\1</span>", escaped)
    escaped = re.sub(r"\b(WARN|WARNING)\b", r"<span class='tok-warn'>\1</span>", escaped)
    escaped = re.sub(r"\b(INFO|DEBUG|NOTICE)\b", r"<span class='tok-info'>\1</span>", escaped)
    escaped = re.sub(r"^(\d{2,4}[/-]\d{2}[/-]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)", r"<span class='tok-ts'>\1</span>", escaped)
    return escaped


def save_parsed_cache(
    filename: str,
    size_bytes: int,
    parsed_entries: list[ParsedLogEntry],
):
    """Caches parsed log data across browser windows."""
    cache_payload = {
        "filename": filename,
        "size_bytes": size_bytes,
        "records": [entry.to_dict() for entry in parsed_entries],
    }
    _SHARED_CACHE["active"] = cache_payload
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f)
    except Exception:
        pass
    return cache_payload


def load_cached_parsed_data():
    """Retrieves cached parsed data across sessions/windows."""
    if "active" in _SHARED_CACHE:
        return _SHARED_CACHE["active"]
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback to the bundled error sample if no parsed data has been cached.
    sample_file = Path(__file__).parent / "sample_error.log"
    if sample_file.exists():
        try:
            content = sample_file.read_text(encoding="utf-8", errors="replace")
            parsed = parse_log_lines(content)
            return {
                "filename": sample_file.name,
                "size_bytes": sample_file.stat().st_size,
                "records": [entry.to_dict() for entry in parsed],
            }
        except Exception:
            pass
    return None


# -----------------------------------------------------------------------------
# VIEW 2: Parsed Log Fields (Opened in New Window / Tab)
# -----------------------------------------------------------------------------
def render_parsed_fields_view():
    data = load_cached_parsed_data()

    # Header with navigation
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                <span style="font-size: 26px;">📊</span>
                <span style="font-size: 24px; font-weight: 700; color: #ffffff;">Parsed Log Fields Explorer</span>
                <span class="pro-badge-ready"><span class="status-dot"></span> Active Window</span>
            </div>
            <div style="color: #94a3b8; font-size: 14px; margin-bottom: 16px;">
                Structured schema extraction (Line Number, Timestamp, Severity Level, Message) from log parser.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_col2:
        st.write("")
        st.link_button("← Return to Console", url="/?view=main", use_container_width=True)

    if not data or not data.get("records"):
        st.warning("⚠️ No active parsed log data found in session. Please upload a log file in the main window.")
        return

    records = data["records"]
    df = pd.DataFrame(records)
    total_count = len(df)
    has_ts_count = df["timestamp"].notna().sum()
    err_count = df["level"].isin(["ERROR", "CRITICAL", "FATAL"]).sum()
    warn_count = df["level"].isin(["WARN", "WARNING"]).sum()

    # KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Parsed Document", data.get("filename", "Log File"), help="Current parsed log source")
    m2.metric("Total Parsed Events", f"{total_count:,}", help="Total structured records")
    m3.metric("Timestamps Identified", f"{has_ts_count:,} ({round((has_ts_count/total_count)*100, 1) if total_count else 0}%)")
    m4.metric(
        "Errors & Criticals",
        f"{err_count:,}",
        delta=f"{warn_count:,} warnings" if warn_count > 0 else None,
        delta_color="inverse",
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Filter Toolbar
    f_col1, f_col2, f_col3 = st.columns([2, 1.2, 1])
    with f_col1:
        search_query = st.text_input(
            "Filter parsed fields:",
            placeholder="🔍 Search in message, timestamp, or level...",
            label_visibility="collapsed",
        )
    with f_col2:
        available_levels = ["All"] + sorted([str(lvl) for lvl in df["level"].dropna().unique()])
        selected_level = st.selectbox("Severity level:", options=available_levels, label_visibility="collapsed")
    with f_col3:
        row_limit = st.selectbox("Rows:", options=[200, 500, 1000, 2000, "All"], index=1, label_visibility="collapsed")

    # Apply Filters
    filtered_df = df.copy()
    if selected_level != "All":
        filtered_df = filtered_df[filtered_df["level"] == selected_level]
    if search_query:
        query_mask = (
            filtered_df["message"].astype(str).str.contains(search_query, case=False, na=False)
            | filtered_df["timestamp"].astype(str).str.contains(search_query, case=False, na=False)
            | filtered_df["level"].astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[query_mask]

    st.caption(f"Showing **{len(filtered_df):,}** of **{total_count:,}** records.")

    # Slice rows
    if row_limit != "All" and len(filtered_df) > row_limit:
        display_df = filtered_df.iloc[:row_limit]
    else:
        display_df = filtered_df

    # Interactive Data Table
    st.dataframe(
        display_df[["line_number", "timestamp", "level", "message"]],
        use_container_width=True,
        column_config={
            "line_number": st.column_config.NumberColumn("Line #", width="small"),
            "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "level": st.column_config.TextColumn("Level", width="small"),
            "message": st.column_config.TextColumn("Parsed Message", width="large"),
        },
        hide_index=True,
        height=480,
    )

    # Exports
    exp_col1, exp_col2 = st.columns([1, 1])
    with exp_col1:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Parsed Fields as CSV",
            data=csv_data,
            file_name=f"parsed_{data.get('filename', 'log')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col2:
        json_data = filtered_df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            label="📥 Export Parsed Fields as JSON",
            data=json_data,
            file_name=f"parsed_{data.get('filename', 'log')}.json",
            mime="application/json",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# VIEW 1: Main Log Console View
# -----------------------------------------------------------------------------
def render_main_view():
    # Hero Header Banner
    st.markdown(
        """
        <div class="pro-header">
            <div class="pro-header-top">
                <div class="pro-title-wrap">
                    <span class="pro-title-icon">🛡️</span>
                    <span class="pro-title-text">AI Log Analysis Assistant</span>
                </div>
                <span class="pro-badge-ready">
                    <span class="status-dot"></span> System Ready
                </span>
            </div>
            <div class="pro-subtitle">
                Automated incident triage, intelligent root-cause detection, and conversational log troubleshooting.
            </div>
            <div class="pro-tags-row">
                <span class="pro-tag">⚡ Automated Error Clustering</span>
                <span class="pro-tag">📊 Severity Analytics</span>
                <span class="pro-tag">🧠 Root Cause Diagnosis</span>
                <span class="pro-tag">💬 Interactive Log Q&A</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File Ingestion Options
    up_col, sample_col = st.columns([2.2, 1.2])
    with up_col:
        uploaded_file = st.file_uploader(
            "Upload Log File",
            type=["log", "txt", "out"],
            help="Supports standard server logs, microservice output, and text trace logs (.log, .txt, .out)",
            label_visibility="collapsed",
        )
    with sample_col:
        sample_options = [
            "-- Or select a sample scenario --",
            "Database Outage (sample_database.log)",
            "Application Crash (sample_application.log)",
            "Authentication Incident (sample_authentication.log)",
            "Network Failure (sample_network.log)",
            "General Errors (sample_error.log)",
        ]
        chosen_sample = st.selectbox(
            "Quick Demo Logs:",
            options=sample_options,
            index=0,
            label_visibility="collapsed",
            help="Choose a pre-loaded sample log to immediately test AI root cause analysis and parsing.",
        )

    # Determine file bytes and filename
    file_bytes = None
    filename = ""

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
    elif chosen_sample != "-- Or select a sample scenario --":
        sample_map = {
            "Database Outage (sample_database.log)": "sample_database.log",
            "Application Crash (sample_application.log)": "sample_application.log",
            "Authentication Incident (sample_authentication.log)": "sample_authentication.log",
            "Network Failure (sample_network.log)": "sample_network.log",
            "General Errors (sample_error.log)": "sample_error.log",
        }
        target_file = Path(__file__).parent / sample_map[chosen_sample]
        if target_file.exists():
            file_bytes = target_file.read_bytes()
            filename = target_file.name

    if file_bytes is None:
        st.markdown(
            """
            <div class="empty-upload-box">
                <div style="font-size: 32px; margin-bottom: 10px;">📄</div>
                <div class="empty-upload-title">Drop your log file above or select a sample scenario</div>
                <div class="empty-upload-sub">
                    Upload any <code>.log</code> or <code>.txt</code> file. Error signatures are automatically extracted and indexed for AI questions.
                </div>
                <div style="margin-top: 18px;">
                    <a href="?view=parsed" target="_blank" class="parsed-hero-btn">
                        <span>📊 Open Parsed Log Fields Explorer</span>
                        <span class="ext-badge">↗ Demo</span>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Process Log Data
    raw_size = len(file_bytes)
    formatted_size = format_file_size(raw_size)
    log_contents = file_bytes.decode("utf-8", errors="replace")
    parsed_entries = parse_log_lines(log_contents)
    total_lines = len(parsed_entries)
    chunks = create_event_chunks(parsed_entries)

    upload_fingerprint = hashlib.sha256(
        filename.encode("utf-8") + b"\0" + file_bytes
    ).hexdigest()

    # Save to shared cache for "Parsed Log Fields" new window
    save_parsed_cache(filename, raw_size, parsed_entries)

    # Quick Counts
    err_count = sum(entry.level in {"ERROR", "CRITICAL", "FATAL"} for entry in parsed_entries)
    warn_count = sum(entry.level in {"WARN", "WARNING"} for entry in parsed_entries)
    err_class = "metric-val-danger" if err_count > 0 else "metric-val-success"

    # Enterprise Metric Cells
    st.markdown(
        f"""
        <div class="metrics-bar">
            <div class="metric-cell">
                <span class="metric-label">Active Document</span>
                <span class="metric-val" title="{filename}">📄 {filename}</span>
            </div>
            <div class="metric-cell">
                <span class="metric-label">File Size</span>
                <span class="metric-val">{formatted_size}</span>
            </div>
            <div class="metric-cell">
                <span class="metric-label">Total Log Lines</span>
                <span class="metric-val">{total_lines:,} lines</span>
            </div>
            <div class="metric-cell">
                <span class="metric-label">Failures Detected</span>
                <span class="metric-val {err_class}">{err_count:,} errors</span>
            </div>
        </div>

        <div class="parsed-banner">
            <div class="parsed-banner-info">
                <div class="parsed-banner-title">
                    <span>📊 Structured Log Schema Ready</span>
                    <span class="parsed-banner-tag">Extracted</span>
                </div>
                <div class="parsed-banner-desc">
                    Extracted <b>{total_lines:,} events</b> into Line Number, Timestamp, Severity Level, and Message schema.
                </div>
            </div>
            <a href="?view=parsed" target="_blank" class="parsed-hero-btn">
                <span>📊 Parsed Log Fields</span>
                <span class="ext-badge">↗ Open in New Window</span>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Index Vector Store for Semantic Search
    index_ready = False
    with st.status("Indexing failure events for AI investigation...", expanded=False) as status:
        vector_store = ChromaVectorStore(
            persistence_path=CACHE_DIR / "chroma_db",
        )
        if chunks:
            try:
                if not vector_store.has_source_chunks(upload_fingerprint, len(chunks)):
                    vector_store.add_chunks(
                        chunks,
                        source=filename,
                        source_id=upload_fingerprint,
                    )
            except GeminiEmbeddingError as error:
                status.update(
                    label="Could not prepare Gemini embeddings for this log.",
                    state="error",
                )
                st.error(str(error))
            else:
                index_ready = True
                status.update(
                    label=(
                        f"✓ Indexed {len(chunks)} failure context chunks. "
                        "Ready for AI questions."
                    ),
                    state="complete",
                )
        else:
            status.update(
                label="Log indexed; no ERROR or CRITICAL entries were found.",
                state="complete",
            )

    # Session State Reset on New File
    if st.session_state.get("semantic_search_source_id") != upload_fingerprint:
        st.session_state["semantic_search_results"] = []
        st.session_state["semantic_search_answer"] = ""
        st.session_state["semantic_search_source_id"] = upload_fingerprint

    # Dual Tab Layout: AI Assistant & Log Console Viewer
    tab_ai, tab_console = st.tabs(
        [
            "🤖 AI Failure Investigation & Root Cause",
            "📜 Raw Log Console & Filter Explorer",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: AI Assistant
    # -------------------------------------------------------------------------
    with tab_ai:
        st.markdown(
            f"""
            <div style="margin-bottom: 12px;">
                <div style="font-size: 17px; font-weight: 700; color: #ffffff;">Ask AI About Log Failures</div>
                <div style="font-size: 13.5px; color: #94a3b8;">
                    Powered by Gemini embeddings and {DEFAULT_GEMINI_MODEL}. Answers cite retrieved log evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not chunks:
            st.info("ℹ️ No ERROR or CRITICAL entries were found in this log to analyze.")

        # Quick Suggestion Pills
        st.caption("Quick Queries:")
        pill_col1, pill_col2, pill_col3, pill_col4 = st.columns(4)
        preset_query = None
        if pill_col1.button("🔥 What caused first error?", use_container_width=True, disabled=not index_ready):
            preset_query = "What caused the first error?"
        if pill_col2.button("🗄️ Database & pool failures?", use_container_width=True, disabled=not index_ready):
            preset_query = "Why did the database connection fail?"
        if pill_col3.button("🛑 Which service crashed?", use_container_width=True, disabled=not index_ready):
            preset_query = "Which component or service failed and why?"
        if pill_col4.button("🛠️ Recommended fixes?", use_container_width=True, disabled=not index_ready):
            preset_query = "What are the recommended next steps to fix these issues?"

        # Search Form
        with st.form("semantic_search_form"):
            default_text = preset_query if preset_query else ""
            query = st.text_input(
                "Enter your question about this log:",
                value=default_text,
                placeholder="e.g. Why did the connection fail? What was the primary root cause?",
                key="semantic_search_query_input",
                disabled=not index_ready,
            )
            form_col1, form_col2 = st.columns([1.5, 4])
            with form_col1:
                search_submitted = st.form_submit_button(
                    "🔍 Ask AI Assistant",
                    disabled=not index_ready,
                    use_container_width=True,
                )
            with form_col2:
                    st.caption(
                        "Set `GOOGLE_API_KEY` in your environment or Streamlit "
                        "secrets to enable Gemini."
                    )

        # Execute Search & Generation if submitted or preset clicked
        active_query = query if search_submitted else (preset_query if preset_query else None)

        if active_query:
            if not active_query.strip():
                st.info("Please enter a question about failures in the uploaded log.")
                st.session_state["semantic_search_results"] = []
                st.session_state["semantic_search_answer"] = ""
            else:
                with st.spinner("Searching relevant log failure chunks and synthesizing root cause..."):
                    st.session_state["semantic_search_results"] = []
                    st.session_state["semantic_search_answer"] = ""
                    try:
                        search_results = vector_store.search(
                            active_query,
                            n_results=5,
                            source_id=upload_fingerprint,
                        )
                    except GeminiEmbeddingError as error:
                        st.error(str(error))
                        search_results = None

                    if search_results is not None:
                        st.session_state["semantic_search_results"] = search_results
                        if not search_results:
                            st.info("No matching failures were found for this query.")
                        else:
                            evidence = [
                                RetrievedEvidence(
                                    source=str(result.metadata["source"]),
                                    start_line_number=int(result.metadata["start_line_number"]),
                                    end_line_number=int(result.metadata["end_line_number"]),
                                    text=result.text,
                                )
                                for result in search_results
                            ]
                            try:
                                answer = generate_log_analysis(
                                    active_query,
                                    evidence,
                                    model_name=DEFAULT_GEMINI_MODEL,
                                )
                            except GeminiError as error:
                                st.error(str(error))
                            else:
                                st.session_state["semantic_search_answer"] = answer

        # Render AI Response
        answer = st.session_state.get("semantic_search_answer", "")
        if answer:
            st.markdown(
                f"""
                <div class="ai-card">
                    <div class="ai-card-title">
                        <span>🧠 AI Root Cause Diagnosis</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(answer)

        # Render Retrieved Evidence
        search_results = st.session_state.get("semantic_search_results", [])
        if search_results:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            with st.expander(f"📑 Retrieved Evidence Chunks ({len(search_results)} matching events)", expanded=False):
                for idx, result in enumerate(search_results, 1):
                    st.markdown(
                        f"**Chunk #{idx}** &bull; <code>{result.metadata['source']}</code> &bull; "
                        f"Lines **{result.metadata['start_line_number']}–{result.metadata['end_line_number']}** &bull; "
                        f"<span style='color: #94a3b8;'>Cosine Distance: {result.distance:.4f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.code(result.text, language="text")

    # -------------------------------------------------------------------------
    # TAB 2: Log Console & Raw Viewer
    # -------------------------------------------------------------------------
    with tab_console:
        all_lines = log_contents.splitlines()

        # Viewer Toolbar: Search, Filters
        tool_col1, tool_col2, tool_col3 = st.columns([2.2, 1.2, 1.1])
        with tool_col1:
            search_kw = st.text_input(
                "Filter log lines:",
                placeholder="🔍 Filter keyword (e.g. exception, timeout, 500)...",
                label_visibility="collapsed",
            )
        with tool_col2:
            filter_mode = st.selectbox(
                "Filter severity:",
                options=["All Lines", "Errors Only", "Warnings Only"],
                index=0,
                label_visibility="collapsed",
            )
        with tool_col3:
            max_lines_choice = st.selectbox(
                "Display limit:",
                options=[150, 300, 600, 1000, "All"],
                index=1,
                label_visibility="collapsed",
            )

        # Apply Filtering
        filtered = []
        for idx, line in enumerate(all_lines, 1):
            if filter_mode == "Errors Only" and not re.search(r"\b(ERROR|CRITICAL|FATAL)\b", line, re.IGNORECASE):
                continue
            if filter_mode == "Warnings Only" and not re.search(r"\b(WARN|WARNING)\b", line, re.IGNORECASE):
                continue
            if search_kw and search_kw.lower() not in line.lower():
                continue
            filtered.append((idx, line))

        filtered_count = len(filtered)

        # Slice for display limit
        if max_lines_choice != "All" and filtered_count > max_lines_choice:
            display_slice = filtered[:max_lines_choice]
            slice_caption = f"Showing first {max_lines_choice} of {filtered_count:,} matching lines"
        else:
            display_slice = filtered
            slice_caption = f"Showing {filtered_count:,} lines"

        # Build Console HTML
        html_rows = []
        for line_num, line_str in display_slice:
            highlighted = highlight_log_line(line_str)
            html_rows.append(f"<div class='log-row'><span class='ln'>{line_num:04d}</span><span class='msg'>{highlighted}</span></div>")

        console_html = "".join(html_rows) if html_rows else "<div style='color: #64748b; padding: 20px; text-align: center;'>No log events match your filter criteria.</div>"

        # Render Professional Console Frame
        st.markdown(
            f"""
            <div class="console-frame">
                <div class="console-topbar">
                    <div class="console-dots">
                        <span class="dot dot-red"></span>
                        <span class="dot dot-yellow"></span>
                        <span class="dot dot-green"></span>
                    </div>
                    <div class="console-meta">
                        {slice_caption} &bull; UTF-8
                    </div>
                </div>
                <div class="console-body">
                    {console_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action Toolbar
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        act_col1, act_col2 = st.columns([1, 1])

        with act_col1:
            st.download_button(
                label="📥 Download Original File",
                data=log_contents,
                file_name=filename,
                mime="text/plain",
                use_container_width=True,
            )

        with act_col2:
            if (search_kw or filter_mode != "All Lines") and filtered:
                filtered_text = "\n".join(item[1] for item in filtered)
                st.download_button(
                    label=f"📥 Download Filtered Output ({filtered_count:,} lines)",
                    data=filtered_text,
                    file_name=f"filtered_{filename}",
                    mime="text/plain",
                    use_container_width=True,
                )


# -----------------------------------------------------------------------------
# Main Routing Entrypoint
# -----------------------------------------------------------------------------
def main():
    if st.query_params.get("view") == "parsed":
        render_parsed_fields_view()
    else:
        render_main_view()


if __name__ == "__main__":
    main()
