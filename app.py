import os
import uuid
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text
from property_details import (
    get_broker,
    get_nearby_facilities_for_property,
    get_property_images,
    get_property_record,
    load_property_catalog,
)

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Chennai Real Estate AI Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) for modern real-estate aesthetics
st.markdown("""
<style>
    /* Global Typography & Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .section-kicker {
        color: #0f766e;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .section-title {
        color: #12343b;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0 0 0.9rem;
    }

    .card-meta {
        color: #52676d;
        font-size: 0.9rem;
        line-height: 1.65;
    }

    .card-description {
        color: #52676d;
        font-size: 0.84rem;
        line-height: 1.5;
        min-height: 3.8rem;
        margin: 0.55rem 0 0.85rem;
    }

    .price-emphasis {
        color: #0f766e;
        font-size: 1.18rem;
        font-weight: 700;
    }

    .detail-panel {
        background: #f8fbfb;
        border: 1px solid #dce9e8;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        min-height: 8rem;
    }

    .detail-label {
        color: #708388;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .detail-value {
        color: #183c43;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    .facility-card {
        background: #ffffff;
        border: 1px solid #e0ebea;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        min-height: 5.4rem;
    }

    .facility-name {
        color: #183c43;
        font-weight: 700;
    }

    .facility-meta {
        color: #64777b;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    /* Hero Banner */
    .hero-container {
        background: #12343b;
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    .hero-icons {
        float: right;
        display: flex;
        gap: 0.8rem;
        color: rgba(255, 255, 255, 0.72);
        font-size: 1.2rem;
    }

    .hero-icons span {
        animation: gentle-float 4s ease-in-out infinite;
    }

    .hero-icons span:nth-child(2) { animation-delay: 0.6s; }
    .hero-icons span:nth-child(3) { animation-delay: 1.2s; }

    @keyframes gentle-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }

    /* Metric Cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* Property Card inside expander */
    .source-card {
        background: #f8fafc;
        border-left: 4px solid #0d9488;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dce9e8;
    }

    /* Pill Badges */
    .tag-pill {
        display: inline-block;
        background: #e2e8f0;
        color: #334155;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .chat-control-row {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        flex-wrap: wrap;
    }
    .chat-control-row .stForm {
        width: 100%;
    }
    .voice-hint {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }

    /* Chat search area */
div[data-testid="stTextInput"] input {
    border-radius: 14px;
    border: 1px solid #d5e1e3;
    min-height: 48px;
    padding-left: 16px;
    font-size: 0.95rem;
}

/* Microphone component */
div[data-testid="stButton"] button {
    border-radius: 14px;
    min-height: 48px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Cached Resources (RAG Engine & Dataset)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag_engine():
    """
    Initializes and caches the RAG chatbot pipeline (FAISS + Sentence Transformers + Gemini).
    """
    from rag_chatbot import RealEstateRAG
    return RealEstateRAG()

@st.cache_data(show_spinner=False)
def load_dataset_metadata():
    """
    Loads dataset statistics for the sidebar.
    """
    path = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
    if not os.path.exists(path):
        path = PROJECT_ROOT / "data" / "chennai_house_price_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_property_catalog_data():
    return load_property_catalog()


def select_property(record):
    property_id = str(record.get("property_id"))
    st.session_state.selected_property = dict(record)
    st.session_state.selected_property_id = property_id
    rag_bot.set_selected_property(record)


def render_property_card(result, key_prefix):
    record = get_property_record(result, load_property_catalog_data())
    property_id = str(record.get("property_id"))
    image_candidates = get_property_images(record)
    _, image_path, _ = image_candidates[0] if image_candidates else ("", "", False)
    with st.container(border=True):
        image_col, detail_col = st.columns([1, 1.7])
        with image_col:
            st.image(str(image_path), use_container_width=True)
        with detail_col:
            st.markdown(f"### {property_id}")
            if record.get("land_type"):
                st.markdown(f"<div class='card-meta'>🌳 {record.get('land_type', 'Property')}<br>📍 {record.get('location', 'Chennai')}<br>📐 {record.get('area_sqft', '—')} sq.ft</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='card-meta'>🏠 {record.get('property_type', 'Property')}<br>📌 {record.get('bhk', '—')} BHK<br>📍 {record.get('location', 'Chennai')}<br>📐 {record.get('area_sqft', '—')} sq.ft</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='price-emphasis'>₹{record.get('price_lakhs', '—')} Lakhs</div>", unsafe_allow_html=True)
            st.button(
                "View Details",
                key=f"view_details_{key_prefix}_{property_id}",
                use_container_width=True,
                on_click=select_property,
                args=(record,),
            )


def render_facility_section(facilities):
    facility_labels = {
        "school": "🏫 Schools", "college": "🎓 Colleges", "hospital": "🏥 Hospitals",
        "supermarket": "🛒 Supermarkets", "transport": "🚌 Transport",
    }
    st.markdown("### Nearby Facilities")
    st.caption("Nearest synthetic/demo records from the current knowledge base.")
    for facility_type, label in facility_labels.items():
        st.markdown(f"#### {label}")
        rows = facilities.get(facility_type, [])
        if not rows:
            st.info("No nearby facilities available in the current knowledge base.")
            continue
        columns = st.columns(min(len(rows), 3))
        for column, row in zip(columns, rows):
            name = row.get(f"{facility_type}_name", row.get("transport_name", "Demo facility"))
            address = row.get("street") or row.get("address") or row.get("area", "Chennai")
            with column:
                st.markdown(
                    f"<div class='facility-card'><div class='facility-name'>{name}</div>"
                    f"<div class='facility-meta'>📍 {row.get('area', 'Chennai')}<br>"
                    f"{address}<br>📏 {row.get('distance_km', 0.0):.1f} km</div></div>",
                    unsafe_allow_html=True,
                )


def render_property_details(record):
    st.markdown("<div class='section-kicker'>Selected listing</div><div class='section-title'>Property Details</div>", unsafe_allow_html=True)
    if st.button("← Back to Results", key=f"close_details_{record.get('property_id')}"):
        st.session_state.selected_property = None
        st.session_state.selected_property_id = None
        rag_bot.clear_selected_property()
        st.rerun()

    if record.get("land_type"):
        st.markdown(f"### 🌳 {record.get('land_type')} · {record.get('property_id')}")
    else:
        st.markdown(f"### 🏠 {record.get('bhk', '—')} BHK {record.get('property_type', 'Property')} · {record.get('property_id')}")
    st.markdown(f"<div class='price-emphasis'>₹{record.get('price_lakhs', '—')} Lakhs</div><div class='card-meta'>📍 {record.get('location', 'Chennai')}, {record.get('city', 'Chennai')} · 📐 {record.get('area_sqft', '—')} sq.ft</div>", unsafe_allow_html=True)
    images = get_property_images(record)
    st.markdown("#### Photos")
    if images:
        st.image(str(images[0][1]), use_container_width=True)
    if len(images) > 1:
        gallery_columns = st.columns(min(len(images), 4))
        for column, (_, image_path, _) in zip(gallery_columns, images):
            with column:
                st.image(str(image_path), use_container_width=True)

    st.markdown("#### Property Overview")
    if record.get("land_type"):
        overview = [("Type", record.get("land_type", "—")), ("Area", f"{record.get('area_sqft', '—')} sq.ft"), ("Price / sq.ft", f"₹{record.get('price_per_sqft', '—')}"), ("Facing", record.get("facing", "—"))]
    else:
        overview = [("Type", record.get("property_type", "—")), ("BHK", record.get("bhk", "—")), ("Bathrooms", record.get("bathrooms", "—")), ("Age", f"{record.get('property_age', '—')} years")]
    for row_start in range(0, len(overview), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, overview[row_start:row_start + 4]):
            with column:
                st.markdown(f"<div class='detail-panel'><div class='detail-label'>{label}</div><div class='detail-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("#### Location")
    location_columns = st.columns(4)
    for column, (label, value) in zip(location_columns, [("Area", record.get("location", "—")), ("Street", record.get("street", "—")), ("City", record.get("city", "Chennai")), ("Pincode", record.get("pincode", "—"))]):
        with column:
            st.markdown(f"<div class='detail-panel'><div class='detail-label'>{label}</div><div class='detail-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("#### Property Information")
    st.markdown(f"**Builder:** {record.get('builder', 'Not specified')}  \n**Property ID:** {record.get('property_id')}  \n\n{record.get('description', 'No description available.')}")

    facilities = get_nearby_facilities_for_property(record)
    render_facility_section(facilities)

    st.markdown("#### Property Contact")
    broker = get_broker(record.get("broker_id"))
    if broker:
        st.info(
            f"DEMO BROKER CONTACT · Fictional project data\n\n"
            f"**{broker['broker_name']}** · {broker['agency_name']}\n\n"
            f"Phone: {broker['phone']}  \nEmail: {broker['email']}  \nAreas served: {broker['areas_served']}"
        )
    else:
        st.info("Demo broker contact is not available for this listing.")


def build_speech_content(answer, sources):
    """Build a voice-friendly summary from the assistant response and any retrieved properties."""
    text = str(answer or "").strip()
    if not text:
        text = "Here are the latest property options for you."

    property_lines = []
    for idx, source in enumerate(sources[:3], start=1):
        meta = source.get("metadata", {}) if isinstance(source, dict) else {}
        source_type = source.get("source_type", meta.get("source_type", "property")) if isinstance(source, dict) else "property"
        if str(source_type).lower() not in {"property", "land"}:
            continue

        location = (
            source.get("location")
            or meta.get("location")
            or meta.get("area")
            or "Chennai"
        )
        bhk = source.get("bhk") or meta.get("bhk")
        property_type = source.get("property_type") or meta.get("property_type")
        price_lakhs = source.get("price_lakhs") or meta.get("price_lakhs") or meta.get("price")

        parts = [f"Property {idx}."]
        if location:
            parts.append(f"Location {location}.")
        if bhk:
            parts.append(f"{bhk} BHK.")
        elif property_type:
            parts.append(f"{property_type}.")
        if price_lakhs:
            try:
                price_value = float(price_lakhs)
                parts.append(f"Price {price_value:.0f} lakhs.")
            except (TypeError, ValueError):
                parts.append(f"Price {price_lakhs}.")
        property_lines.append(" ".join(parts))

    if property_lines:
        text = f"{text}\n\n" + "\n".join(property_lines)

    return text

# Verify environment configuration
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Sorry, the property assistant is temporarily unavailable. Please try again shortly.")
    st.stop()

# Initialize RAG Pipeline
try:
    with st.spinner("⚡ Initializing FAISS Vector Store & AI Models..."):
        rag_bot = load_rag_engine()
        df_stats = load_dataset_metadata()
except Exception:
    st.error("Sorry, the property assistant is temporarily unavailable. Please try again shortly.")
    st.stop()

# -------------------------------------------------------------
# Sidebar: System Metrics & Controls
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/real-estate.png", width=70)
    st.markdown("### 🏢 Knowledge Base Stats")

    location_data_path = PROJECT_ROOT / "data" / "locations.csv"
    location_placeholder = "Select an area"
    if location_data_path.exists():
        location_options = [location_placeholder] + sorted(pd.read_csv(location_data_path)["area"].dropna().astype(str).str.strip().unique().tolist())
        current_location = rag_bot.conversation_context.get("current_location")
        if "location_selector" not in st.session_state:
            st.session_state.location_selector = current_location or location_placeholder
        elif current_location and st.session_state.location_selector != current_location:
            st.session_state.location_selector = current_location
        elif not current_location:
            st.session_state.location_selector = location_placeholder

        def update_selected_location():
            selected_area = st.session_state.location_selector
            if selected_area == location_placeholder:
                rag_bot.clear_current_location()
            else:
                rag_bot.set_current_location(selected_area)

        st.selectbox("Current Search Location", location_options, key="location_selector", on_change=update_selected_location)
        st.caption("Nearby searches include areas defined in locations.csv.")

    if not df_stats.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df_stats):,}</div>
                <div class="metric-label">Properties</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df_stats['location'].nunique()}</div>
                <div class="metric-label">Locations</div>
            </div>
            """, unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{df_stats['price'].min():.1f}L</div>
                <div class="metric-label">Min Price</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{df_stats['price'].max():.0f}L</div>
                <div class="metric-label">Max Price</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("---")
    st.markdown("### 💡 Quick Search Prompts")
    quick_prompts = [
        "3 BHK properties in Anna Nagar",
        "Properties under 50 lakhs",
        "Ready to move 2 BHK in Sembakkam",
        "Which properties are available?",
        "Luxury properties above 200 lakhs"
    ]

    for qp in quick_prompts:
        if st.button(f"🔍 {qp}", key=f"quick_{qp}", use_container_width=True):
            st.session_state.pending_query = qp

    st.markdown("---")
    def clear_chat_and_location():
        st.session_state.messages = []
        rag_bot.clear_memory()
        st.session_state.location_selector = location_placeholder
        st.session_state.selected_property = None
        st.session_state.selected_property_id = None
        st.session_state.pop("pending_query", None)

    st.button(
        "🗑️ Clear Chat & Memory",
        width="stretch",
        type="secondary",
        on_click=clear_chat_and_location,
    )

# -------------------------------------------------------------
# Main Chat Application Interface
# -------------------------------------------------------------
# Header
st.markdown("""
<div class="hero-container">
    <div class="hero-icons" aria-hidden="true"><span>🏠</span><span>🔑</span><span>📍</span></div>
    <span class="hero-badge">Chennai Real Estate Search</span>
    <h1 class="hero-title">Chennai Property AI Assistant</h1>
    <p class="hero-subtitle">Find properties, explore nearby facilities and get property insights.</p>
</div>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_retrieved_properties" not in st.session_state:
    st.session_state.last_retrieved_properties = []
if "selected_property" not in st.session_state:
    st.session_state.selected_property = None
if "selected_property_id" not in st.session_state:
    st.session_state.selected_property_id = None
if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""

selected_property = st.session_state.get("selected_property")
selected_property_id = st.session_state.get("selected_property_id")
if selected_property_id:
    catalog = load_property_catalog_data()
    selected_matches = catalog[catalog["property_id"].astype(str) == str(selected_property_id)]
    if not selected_matches.empty:
        selected_property = selected_matches.iloc[0].to_dict()
        st.session_state.selected_property = selected_property

if selected_property:
    with st.container(border=True):
        render_property_details(selected_property)
    st.stop()

# Display Conversation History
for message_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(
        msg["role"],
        avatar="🧑‍💼" if msg["role"] == "assistant" else "👤"
    ):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":

            # Use the complete speech text if available.
            # Otherwise use the normal assistant message.
            speech_text = msg.get("speech_content", msg["content"])

            active_speech = (
                st.session_state.get("voice_output_text") == speech_text
            )

            is_speaking = (
                st.session_state.get("speech_state") == "speaking"
                and active_speech
            )

            if st.button(
                "🔊 ✕ Stop" if is_speaking else "🔊 Listen",
                key=f"listen_{message_index}",
                width="content"
            ):

                if is_speaking:

                    st.session_state.voice_output_text = None
                    st.session_state.speech_state = "stopped"
                    st.session_state.speech_token = uuid.uuid4().hex

                    components.html(
                        """
                        <script>
                            if (window.speechSynthesis) {
                                window.speechSynthesis.cancel();
                            }
                        </script>
                        """,
                        height=0,
                    )

                else:

                    st.session_state.voice_output_text = speech_text
                    st.session_state.speech_state = "speaking"
                    st.session_state.speech_token = uuid.uuid4().hex

                        # Speak the selected assistant response
            if active_speech and st.session_state.get("speech_state") == "speaking":

                token = st.session_state.get("speech_token") or ""

                speech_text = msg.get(
                    "speech_content",
                    msg.get("content", "")
                )

                components.html(
                    f"""
                    <script>
                        (() => {{
                            const token = {token!r};
                            const text = {speech_text!r};

                            if (!('speechSynthesis' in window)) {{
                                return;
                            }}

                            const lastToken =
                                window.__realEstateSpeechToken || '';

                            if (lastToken === token) {{
                                return;
                            }}

                            window.speechSynthesis.cancel();

                            const utterance =
                                new SpeechSynthesisUtterance(text);

                            utterance.lang = 'en-US';
                            utterance.rate = 1;

                            utterance.onend = () => {{
                                window.__realEstateSpeechToken = token;

                                if (
                                    window.parent &&
                                    window.parent !== window
                                ) {{
                                    try {{
                                        window.parent.postMessage(
                                            {{
                                                type: 'speech-end',
                                                token: token
                                            }},
                                            '*'
                                        );
                                    }} catch (e) {{}}
                                }}
                            }};

                            window.__realEstateSpeechToken = token;

                            window.speechSynthesis.speak(utterance);
                        }})();
                    </script>
                    """,
                    height=0,
                )

    if msg.get("sources"):
            with st.expander(f"🔍 View {len(msg['sources'])} Retrieved FAISS Knowledge Base Sources"):
                property_sources = [source for source in msg["sources"] if source.get("source_type", source.get("metadata", {}).get("source_type", "property")) in {"property", "land"}]
                if property_sources:
                    st.markdown("#### Retrieved Properties")
                    for property_index, source in enumerate(property_sources):
                        render_property_card(source, f"history_{message_index}_{property_index}")
                for s in msg["sources"]:
                    meta = s.get("metadata", {})
                    source_type = s.get("source_type", meta.get("source_type", "property")).title()
                    source_id = s.get("source_id", meta.get("source_id", s.get("property_id")))
                    source_area = meta.get("location", meta.get("area", "Chennai"))
                    if source_type.lower() in {"property", "land"}:
                        continue
                    st.markdown(f"""
                    <div class="source-card">
                        <b>{source_type} ID {source_id}</b> — <span class="tag-pill">Cosine Similarity: {s.get('similarity', 0.0):.3f}</span><br>
                        <b>Location:</b> {source_area} | <b>Type:</b> {meta.get('property_type', meta.get('school_type', meta.get('college_type', meta.get('hospital_type', meta.get('transport_type', '')))))}<br>
                        <b>Details:</b> {meta.get('description', 'Retrieved from the knowledge base.')}<br>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("<div class='section-kicker'>Search</div><div class='section-title'>Find your next Chennai property</div>", unsafe_allow_html=True)
st.caption("Try a natural-language search such as ‘3 BHK in Velachery’ or ‘land in Tambaram’.")

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# -------------------------------------------------------------
# Chat Search Bar + Voice Input
# -------------------------------------------------------------

# Initialize search states BEFORE creating any widgets
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "pending_voice_query" not in st.session_state:
    st.session_state.pending_voice_query = None

if "last_voice_query" not in st.session_state:
    st.session_state.last_voice_query = ""

if "submitted_query" not in st.session_state:
    st.session_state.submitted_query = None

if "clear_search_after_submit" not in st.session_state:
    st.session_state.clear_search_after_submit = False


# -------------------------------------------------------------
# Prepare state BEFORE the search widget is created
# -------------------------------------------------------------

# Put new voice text into the search box BEFORE creating
# the text input widget.
if st.session_state.pending_voice_query:
    st.session_state.search_query = st.session_state.pending_voice_query
    st.session_state.pending_voice_query = None


# Clear the search box on the next rerun after sending
if st.session_state.clear_search_after_submit:
    st.session_state.search_query = ""
    st.session_state.clear_search_after_submit = False


# -------------------------------------------------------------
# Send button callback
# -------------------------------------------------------------

def submit_search():
    query = st.session_state.search_query.strip()

    if query:
        # Store query separately
        st.session_state.submitted_query = query

        # Tell the next rerun to clear the visible input
        st.session_state.clear_search_after_submit = True


# -------------------------------------------------------------
# Search Row
# -------------------------------------------------------------

search_col, voice_col, send_col = st.columns(
    [10, 0.9, 0.9],
    vertical_alignment="bottom"
)


# -------------------------------------------------------------
# Text Search Box
# -------------------------------------------------------------

with search_col:
    st.text_input(
        "Search",
        key="search_query",
        placeholder="Ask about properties in Chennai",
        label_visibility="collapsed"
    )


# -------------------------------------------------------------
# Voice Button
# -------------------------------------------------------------

with voice_col:
    voice_query = speech_to_text(
        language="en",
        start_prompt="🎤",
        stop_prompt="⏹️",
        just_once=True,
        use_container_width=True,
        key="real_estate_voice"
    )


# -------------------------------------------------------------
# Handle Voice Result
# -------------------------------------------------------------

if voice_query:

    # Only process a NEW voice transcription
    if voice_query != st.session_state.last_voice_query:

        st.session_state.last_voice_query = voice_query

        # Do NOT modify search_query here.
        # Store it for the next Streamlit rerun instead.
        st.session_state.pending_voice_query = voice_query

        st.rerun()


# -------------------------------------------------------------
# Send Button
# -------------------------------------------------------------

with send_col:
    st.button(
        "↑",
        key="send_property_query",
        on_click=submit_search,
        width="stretch"
    )

# -------------------------------------------------------------
# Process Submitted Query
# -------------------------------------------------------------

user_input = None

# Text box / Send button
if st.session_state.get("submitted_query"):
    user_input = st.session_state.submitted_query
    st.session_state.submitted_query = None

# Quick search buttons
elif st.session_state.get("pending_query"):
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None


# -------------------------------------------------------------
# Process the user's question
# -------------------------------------------------------------

if user_input:

    user_input = str(user_input).strip()

    if user_input:

        # Add user message to visible chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "sources": []
        })

        try:
            # IMPORTANT:
            # Use the SAME cached RAG bot so that its
            # conversation_history is preserved.
            answer = rag_bot.ask(user_input)

        except Exception as e:
            print(f"RAG error: {e}")

            answer = (
                "Sorry, I couldn't process that question right now. "
                "Please try again."
            )

        # Get properties retrieved for this question
        sources = list(
            getattr(
                rag_bot,
                "last_retrieved_properties",
                []
            ) or []
        )

        # Create voice-friendly response
        speech_content = build_speech_content(
            answer,
            sources
        )

        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "speech_content": speech_content,
            "sources": sources
        })

        # Keep Streamlit state synchronized
        st.session_state.last_retrieved_properties = sources
        st.session_state.last_search_query = user_input

        # Keep a copy of the RAG conversation history
        st.session_state.conversation_history = list(
            getattr(
                rag_bot,
                "conversation_history",
                []
            )
        )

        # Clear selected property when starting a reset command
        reset_commands = {
            "new search",
            "clear conversation",
            "clear chat",
            "clear",
            "reset",
            "reset memory"
        }

        if user_input.lower().rstrip("?!. ") in reset_commands:

            st.session_state.selected_property = None
            st.session_state.selected_property_id = None

            # Also clear the RAG memory
            rag_bot.clear_memory()

        # Rerun only AFTER the complete answer has been saved
        st.rerun()
 
        
# Footer# 

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "🏠 Real-Estate-RAG | Powered by FAISS, Sentence Transformers (all-MiniLM-L6-v2), and Google Gemini 3.6 Flash"
    "</p>",
    unsafe_allow_html=True
)