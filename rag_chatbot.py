import os
import re
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Common word to number mapping for BHK and other price-related natural language queries
WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

GREETINGS = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", "namaste", "help", "who are you", "thanks", "thank you", "how are you"}
PROPERTY_SEARCH_HINTS = (
    "property", "properties", "house", "houses", "apartment", "apartments", "villa", "villas",
    "flat", "flats", "land", "plot", "plots", "bhk", "bedroom", "bedrooms", "price",
    "lakhs", "lakh", "crore", "crores", "budget", "under", "above", "within", "near",
    "location", "in ", "for sale", "for rent"
)
OUT_OF_SCOPE_LOCATIONS = {"coimbatore", "madurai", "tiruchirappalli", "trichy", "salem", "bengaluru", "bangalore", "hyderabad"}
UNGROUNDED_ATTRIBUTE_TERMS = {
    "school quality", "crime", "safety", "loan", "interest", "down payment",
    "walking distance", "approval", "investment return"
}

SOURCE_KEYWORDS = {
    "property": ("property", "apartment", "house", "villa", "bhk"),
    "land": ("land", "plot"),
    "school": ("school",),
    "hospital": ("hospital",),
    "college": ("college",),
    "supermarket": ("supermarket", "market"),
    "transport": ("transport", "bus stop", "railway", "metro", "auto stand"),
    "broker": ("broker", "contact", "phone", "email"),
}

def normalize_query_text(query: str) -> str:
    """
    Normalizes natural language queries into a consistent form for filtering,
    such as 'three BHK' -> '3 bhk' and 'under 50 lakhs' -> 'under 50 lakhs'.
    """
    norm = (query or "").lower().strip()
    norm = norm.replace("₹", "rs ")
    norm = norm.replace("inr", "rs")
    norm = re.sub(r"[^a-z0-9\s.]+", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()

    for word, num in WORD_TO_NUM.items():
        norm = re.sub(rf'\b{word}\s*(?:bhk|bedroom|bedrooms|bed)\b', f'{num} bhk', norm)

    # Normalize simple conversational phrases without damaging the original structure
    norm = re.sub(r'\bthree\s+bhk\b', '3 bhk', norm)
    norm = re.sub(r'\bthree\s+bedroom\b', '3 bhk', norm)
    norm = re.sub(r'\b(?:around|near|nearby)\s+(?:the\s+)?', '', norm)
    return norm

def extract_query_filters(query: str):
    """
    Extracts structured constraints (BHK, max price, location) from natural language queries.
    Keeps compatibility with the existing filters while accepting common phrasing variations.
    """
    norm = normalize_query_text(query)
    filters = {"bhk": None, "min_price": None, "max_price": None, "location": None}

    # Extract BHK (e.g., "3 bhk", "2 bedroom", "three BHK")
    bhk_match = re.search(r'(\d+)\s*(?:bhk|bedroom|bedrooms|bed)\b', norm)
    if bhk_match:
        filters["bhk"] = int(bhk_match.group(1))

    # Extract price constraints (e.g., "under 50 lakhs", "below 75l", "less than 1 crore")
    price_match = re.search(
        r'(?:under|below|less than|not more than|maximum|max|within|upto|up to|budget\s*(?:is|of)?\s*)\s*(?:rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|l|crore|crores|cr)?\b',
        norm
    )
    if price_match:
        val = float(price_match.group(1).replace(",", ""))
        unit = (price_match.group(2) or "").lower()
        if unit in ["crore", "crores", "cr"]:
            filters["max_price"] = val * 100.0
        elif unit in ["lakh", "lakhs", "lac", "lacs", "l"]:
            filters["max_price"] = val
        else:
            if val >= 100000:
                filters["max_price"] = val / 100000.0
            else:
                filters["max_price"] = val

    min_price_match = re.search(
        r'(?:above|over|at least|minimum|min)\s*(?:rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|l|crore|crores|cr)?\b',
        norm
    )
    if min_price_match:
        val = float(min_price_match.group(1).replace(",", ""))
        unit = (min_price_match.group(2) or "").lower()
        if unit in ["crore", "crores", "cr"]:
            filters["min_price"] = val * 100.0
        elif unit in ["lakh", "lakhs", "lac", "lacs", "l"]:
            filters["min_price"] = val
        else:
            filters["min_price"] = val / 100000.0 if val >= 100000 else val

    return filters

def is_broad_property_query(query: str, filters: dict) -> bool:
    """
    Checks if the user has expressed a general intent to look for properties
    without specifying concrete constraints (location, BHK, price) yet.
    """
    q = (query or "").lower().strip("?!. ")
    if any(filters.get(k) is not None for k in ("location", "bhk", "min_price", "max_price")):
        return False
    broad_patterns = [
        r"\b(looking for a property|looking for property|look for a property|search for a property|need a property|want a property|find a property|find me a property)\b",
        r"\b(looking for a house|need a house|want a house|looking for an apartment|need an apartment|want an apartment|looking for a flat|need a flat)\b",
        r"\b(looking for land|need land|want land|looking for plot|need plot)\b",
        r"\b(help me find a property|help me find a home|help me find a house)\b",
        r"\b(i want to buy a house|i want to buy a property|i want to buy an apartment|i am looking for a property|i'm looking for a property)\b",
        r"^(?:i am|i'm|im)?\s*(?:looking for|searching for|want|need)\s+(?:a\s+)?(?:property|house|apartment|home|land|plot)s?$",
    ]
    return any(re.search(pat, q) for pat in broad_patterns)


def is_property_search_query(query: str) -> bool:
    """Return True only when the user's message asks for real estate/property matching."""
    if query is None:
        return False
    q = normalize_query_text(str(query)).strip().lower()
    if not q:
        return False
    if q in GREETINGS or re.search(r"^(?:hi|hello|hey|good morning|good afternoon|good evening|namaste|thanks|thank you|how are you|howdy)\b", q):
        return False
    if any(q.startswith(prefix) for prefix in ("what is your", "who are you", "are you", "can you", "please")):
        return False
    if re.search(r"\b(?:under|below|less than|budget|upto|up to|maximum|max|minimum|min|above|over|not more than)\b.*\b(?:lakhs?|crore|crores|cr)\b", q):
        return True
    if re.search(r"\b\d+\s*(?:bhk|bedroom|bedrooms|bed)\b", q):
        return True
    if re.search(r"\b(?:property|properties|house|houses|apartment|apartments|villa|villas|flat|flats|land|plot|plots)\b", q):
        return True
    if re.search(r"\b(?:in|near|around|nearby)\s+[a-z]+", q):
        return True
    if any(hint in q for hint in ("lakhs", "lakh", "crore", "crores", "budget", "price", "location", "bhk", "property", "house", "apartment", "villa", "land", "plot")):
        return True
    return False


def extract_property_selection(query: str, last_items: list):
    """
    Checks if query refers to viewing/selecting a specific property by number, ordinal, or ID.
    Examples:
      - 'Tell me about property 5'
      - 'What about the fifth one?'
      - 'I want to see property 5'
      - 'Show me property 2'
      - 'View details for property 3'
      - 'Property 5'
    """
    q = (query or "").lower().strip("?!. ")

    # Check code match: PROP005, LAND001, etc.
    code_match = re.search(r'\b(prop\d{3}|land\d{3}|legacy-prop-\d{4})\b', q, re.I)
    if code_match:
        code = code_match.group(1).upper()
        if last_items:
            for i, item in enumerate(last_items):
                if str(item.get("source_id", "")).upper() == code or str(item.get("property_id", "")).upper() == code:
                    return item, i
        from property_details import load_property_catalog
        catalog = load_property_catalog()
        matches = catalog[catalog["property_id"].astype(str) == code]
        if not matches.empty:
            return {"source_id": code, "metadata": matches.iloc[0].to_dict()}, 0

    # Check digit match: "property 5", "prop 5", "property #5", "#5"
    m = re.search(r'\b(?:property|prop|option|listing|#)\s*(\d+)\b', q)
    if not m and re.search(r'^(?:show\s+me\s+|view\s+|open\s+|tell\s+me\s+about\s+)?(\d+)$', q):
        m = re.search(r'(\d+)', q)
    if m:
        val = int(m.group(1))
        # Ensure it's not a BHK or price (e.g. "3 bhk", "60 lakhs")
        if not re.search(rf'\b{val}\s*(?:bhk|bedroom|bedrooms|bed|lakh|lakhs|cr|crore|sqft|sq\.?\s*ft)\b', q):
            idx = val - 1
            if last_items and 0 <= idx < len(last_items):
                return last_items[idx], idx
            from property_details import load_property_catalog
            catalog = load_property_catalog()
            prop_id = f"PROP{val:03d}"
            matches = catalog[catalog["property_id"].astype(str) == prop_id]
            if not matches.empty:
                return {"source_id": prop_id, "metadata": matches.iloc[0].to_dict()}, idx

    # Check ordinal word match: "first", "second", "third", "fourth", "fifth", etc.
    ord_map = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3,
        "fifth": 4, "5th": 4,
        "sixth": 5, "6th": 5,
        "seventh": 6, "7th": 6,
        "eighth": 7, "8th": 7,
        "ninth": 8, "9th": 8,
        "tenth": 9, "10th": 9,
    }
    for word, idx in ord_map.items():
        if re.search(rf'\b{word}\s*(?:one|property|listing|option)?\b', q):
            if last_items and 0 <= idx < len(last_items):
                return last_items[idx], idx
            from property_details import load_property_catalog
            catalog = load_property_catalog()
            prop_id = f"PROP{idx + 1:03d}"
            matches = catalog[catalog["property_id"].astype(str) == prop_id]
            if not matches.empty:
                return {"source_id": prop_id, "metadata": matches.iloc[0].to_dict()}, idx

    return None, None

class RealEstateRAG:
    """
    Production-grade RAG pipeline for Chennai Real Estate.
    Uses FAISS for semantic vector search, Sentence Transformers for embeddings,
    and Gemini 3.6 Flash for grounded, hallucination-free generation with conversation memory.
    """



    def __init__(self, index_dir="faiss_index", dataset_path=None, model_name="all-MiniLM-L6-v2", require_gemini=True):
        self.index_dir = self._resolve_project_path(index_dir)
        self.model_name = model_name
        self.gemini_model = "gemini-3.6-flash"

        # Initialize Gemini API Client
        api_key = os.getenv("GEMINI_API_KEY")
        if require_gemini and not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file!")
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

        # Load embedding model
        print(f"Loading embedding model ({self.model_name})...")
        self.embedding_model = SentenceTransformer(self.model_name)

        # Load FAISS index & metadata
        self.load_faiss()

        # Load tabular dataset for hybrid lookup and location detection
        if dataset_path is None:
            dataset_path = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
            if not os.path.exists(dataset_path):
                dataset_path = PROJECT_ROOT / "data" / "chennai_house_price_cleaned.csv"
        self.dataset_path = Path(dataset_path)
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path)
            self.locations = [loc.strip().lower() for loc in self.df["location"].dropna().unique()]
        else:
            self.df = pd.DataFrame()
            self.locations = []
        location_path = PROJECT_ROOT / "data" / "locations.csv"
        self.location_records = {}
        if location_path.exists():
            location_df = pd.read_csv(location_path)
            for _, row in location_df.iterrows():
                area = str(row["area"]).strip()
                nearby = [value.strip() for value in str(row["nearby_areas"]).split(";") if value.strip()]
                self.location_records[area.lower()] = {"area": area, "nearby_areas": nearby}
            self.locations = sorted(set(self.locations) | set(self.location_records))

        # Conversation memory: stores list of {"role": "user"|"assistant", "content": "..."}
        self.conversation_history = []
        self.last_retrieved_properties = []
        self.shown_property_ids = set()
        self.selected_property_context = None
        self.conversation_context = {
            "location": None,
            "current_location": None,
            "nearby_locations": [],
            "property_type": None,
            "bhk": None,
            "min_price": None,
            "max_price": None,
            "area_sqft": None,
            "land_requirement": None,
            "nearby_requirement": None,
            "facility_type": None,
        }

    def handle_followup_comparison(self, user_input):
        """
        Handles follow-up questions such as:
        - which property is cheaper?
        - which one is expensive?
        - which property costs less?
        - compare these properties
        """

        query = user_input.lower().strip()

        comparison_words = [
            "cheaper",
            "cheapest",
            "lower price",
            "less expensive",
            "costs less",
            "cheapest property",
            "compare",
            "comparison",
            "expensive",
            "highest price",
            "costs more"
        ]

        # Check whether this is a comparison/follow-up question
        if not any(word in query for word in comparison_words):
            return None

        # Get properties from the previous search
        previous_properties = getattr(self, "last_retrieved_properties", [])

        if not previous_properties:
            return (
                "I don't have the previous property search available. "
                "Please search for some properties first."
            )

        # Extract valid property prices
        properties_with_price = []

        for property_data in previous_properties:
            try:
                price = float(property_data.get("price", 0))
                properties_with_price.append((price, property_data))
            except (ValueError, TypeError):
                continue

        if not properties_with_price:
            return "I couldn't find valid property prices to compare."

        # Cheapest property
        if any(word in query for word in [
            "cheaper",
            "cheapest",
            "lower price",
            "less expensive",
            "costs less"
        ]):
            cheapest_price, cheapest_property = min(
                properties_with_price,
                key=lambda x: x[0]
            )

            location = cheapest_property.get("location", "Unknown location")
            bhk = cheapest_property.get("bhk", "Unknown")
            property_id = cheapest_property.get("property_id", "Unknown")

            return (
                f"The cheaper property is Property ID {property_id} "
                f"in {location}. It is a {bhk} property priced at "
                f"₹{cheapest_price:.2f} Lakhs."
            )

        # Most expensive property
        if any(word in query for word in [
            "expensive",
            "highest price",
            "costs more"
        ]):
            expensive_price, expensive_property = max(
                properties_with_price,
                key=lambda x: x[0]
            )

            location = expensive_property.get("location", "Unknown location")
            bhk = expensive_property.get("bhk", "Unknown")
            property_id = expensive_property.get("property_id", "Unknown")

            return (
                f"The most expensive property is Property ID {property_id} "
                f"in {location}. It is a {bhk} property priced at "
                f"₹{expensive_price:.2f} Lakhs."
            )

        # General comparison
        sorted_properties = sorted(
            properties_with_price,
            key=lambda x: x[0]
        )

        comparison = "Here is the price comparison:\n\n"

        for index, (price, property_data) in enumerate(
            sorted_properties,
            start=1
        ):
            property_id = property_data.get("property_id", "Unknown")
            location = property_data.get("location", "Unknown location")

            comparison += (
                f"{index}. Property ID {property_id} — "
                f"{location} — ₹{price:.2f} Lakhs\n"
            )

        return comparison

    @staticmethod
    def _resolve_project_path(path):
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def load_faiss(self):
        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.pkl"

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"FAISS index files not found in '{self.index_dir}'! "
                f"Please run 'python create_vector_db.py' to generate the index."
            )

        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.metadata_payload = pickle.load(f)

        self.documents = self.metadata_payload["documents"]
        self.meta_list = self.metadata_payload.get("metadata", [])
        print(f"FAISS index loaded successfully with {self.index.ntotal} documents.")

    @staticmethod
    def detect_source_type(query: str):
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in SOURCE_KEYWORDS["broker"]):
            return "broker"
        for source_type, keywords in SOURCE_KEYWORDS.items():
            if source_type == "broker":
                continue
            if any(keyword in query_lower for keyword in keywords):
                return source_type
        return None

    def detect_location(self, query: str):
        """
        Detects if a known Chennai location is mentioned in the query.
        """
        norm = query.lower()
        for loc in self.locations:
            if re.search(rf'\b{re.escape(loc)}\b', norm):
                return loc.title()
        return None

    def set_current_location(self, area):
        """Set a validated current area and load its dataset-defined nearby areas."""
        if not area:
            return None
        record = self.location_records.get(str(area).strip().lower())
        if record is None:
            raise ValueError(f"Unknown Chennai area: {area}")
        current_area = record["area"]
        nearby_locations = [current_area] + [value for value in record["nearby_areas"] if value.lower() != current_area.lower()]
        self.conversation_context["location"] = current_area
        self.conversation_context["current_location"] = current_area
        self.conversation_context["nearby_locations"] = nearby_locations
        return current_area

    def clear_current_location(self):
        """Clear location selection without clearing other conversation filters."""
        self.conversation_context["location"] = None
        self.conversation_context["current_location"] = None
        self.conversation_context["nearby_locations"] = []

    def get_search_locations(self):
        """Return the current area and its validated nearby areas."""
        return list(self.conversation_context.get("nearby_locations") or [])

    def set_selected_property(self, property_record):
        """Keep the selected property available for property-specific follow-ups."""
        self.selected_property_context = dict(property_record)
        location = property_record.get("location")
        if location and str(location).lower() in self.location_records:
            self.set_current_location(location)

    def clear_selected_property(self):
        self.selected_property_context = None

    def extract_conversation_context(self, query: str):
        """Extract only requirements explicitly stated in the current message."""
        norm_query = normalize_query_text(query)
        filters = extract_query_filters(norm_query)
        context = {}

        if filters["location"] is not None:
            context["location"] = filters["location"]
        else:
            detected_location = self.detect_location(norm_query)
            if detected_location:
                context["location"] = detected_location
        for key in ("bhk", "min_price", "max_price"):
            if filters[key] is not None:
                context[key] = filters[key]

        area_match = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|square feet)", norm_query)
        if area_match:
            context["area_sqft"] = float(area_match.group(1).replace(",", ""))

        if re.search(r"\b(independent\s+house|house|houses|home|homes)\b", norm_query):
            context["property_type"] = "house"
        elif re.search(r"\b(apartment|apartments|flat|flats)\b", norm_query):
            context["property_type"] = "apartment"
        elif re.search(r"\bvilla|villas\b", norm_query):
            context["property_type"] = "villa"
        elif re.search(r"\bproperty|properties\b", norm_query):
            context["property_type"] = "property"

        if re.search(r"\b(land|plot|plots)\b", norm_query):
            context["land_requirement"] = True
        if re.search(r"\b(near|nearby|nearest|around|around there)\b", norm_query):
            context["nearby_requirement"] = True

        facility_patterns = {
            "school": r"\bschools?\b",
            "college": r"\bcolleges?\b",
            "hospital": r"\bhospitals?\b",
            "supermarket": r"\bsupermarkets?|markets?\b",
            "transport": r"\btransport|bus stops?|railway stations?|metro stations?|auto stands?\b",
        }
        for facility_type, pattern in facility_patterns.items():
            if re.search(pattern, norm_query):
                context["facility_type"] = facility_type
                break
        return context

    def update_conversation_context(self, query: str):
        """Merge explicitly stated values into the existing structured context."""
        updates = self.extract_conversation_context(query)
        if "location" in updates:
            self.set_current_location(updates.pop("location"))
        for key, value in updates.items():
            self.conversation_context[key] = value
        return dict(self.conversation_context)

    def build_contextual_query(self, query: str):
        """Combine the current request with known context without inventing values."""
        context = self.update_conversation_context(query)
        context_parts = []
        if context["location"]:
            context_parts.append(f"location {context['location']}")
        if context["nearby_requirement"] and context["nearby_locations"]:
            context_parts.append("search areas " + ", ".join(context["nearby_locations"]))
        if context["property_type"]:
            context_parts.append(context["property_type"])
        if context["bhk"] is not None:
            context_parts.append(f"{context['bhk']} BHK")
        if context["min_price"] is not None:
            context_parts.append(f"minimum price {context['min_price']} lakhs")
        if context["max_price"] is not None:
            context_parts.append(f"maximum price {context['max_price']} lakhs")
        if context["area_sqft"] is not None:
            context_parts.append(f"area {context['area_sqft']} sq.ft")
        if context["land_requirement"]:
            context_parts.append("land")
        if context["nearby_requirement"]:
            context_parts.append("nearby")
        if context["facility_type"]:
            context_parts.append(context["facility_type"])
        selected = self.selected_property_context
        if selected and re.search(r"\b(this property|this house|this land|around here|nearby|nearest|about this property)\b", query.lower()):
            context_parts.append(f"selected property {selected.get('property_id', selected.get('source_id'))}")
            if selected.get("location"):
                context_parts.append(f"at {selected['location']}")
        if not context_parts:
            return query
        return f"{query} ({', '.join(context_parts)})"

    def needs_location_for_nearby(self):
        return bool(self.conversation_context.get("nearby_requirement") and not self.conversation_context.get("location"))

    def retrieve(self, query: str, top_k=5, min_similarity=0.22, location_scope=None, exclude_ids=None):
        """
        Hybrid retrieval combining FAISS semantic vector search with constraint filtering.
        Supports exclude_ids for pagination / 'show more' queries.
        """
        norm_query = normalize_query_text(query)
        if any(re.search(rf"\b{re.escape(city)}\b", norm_query) for city in OUT_OF_SCOPE_LOCATIONS):
            return [], {"bhk": None, "max_price": None, "location": None}
        filters = extract_query_filters(norm_query)
        detected_loc = self.detect_location(norm_query)
        if detected_loc:
            filters["location"] = detected_loc
        requested_source_type = self.detect_source_type(norm_query)

        exclude_set = {str(e).strip().lower() for e in exclude_ids} if exclude_ids else set()

        # 1. FAISS Semantic Search
        query_emb = self.embedding_model.encode([norm_query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_emb)

        # Retrieve a broader pool of candidates from FAISS for filtering
        search_k = min(max((top_k + len(exclude_set)) * 4, 30), self.index.ntotal)
        scores, indices = self.index.search(query_emb, search_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            meta = self.meta_list[idx] if idx < len(self.meta_list) else {}
            candidates.append({
                "property_id": idx + 1,
                "source_type": meta.get("source_type", "property"),
                "source_id": meta.get("source_id", idx + 1),
                "document": self.documents[idx],
                "similarity": float(score),
                "metadata": meta
            })

        has_constraints = any(v is not None for v in filters.values())

        if has_constraints:
            # Filter candidates based on detected constraints
            filtered = []
            for item in candidates:
                meta = item["metadata"]
                match = True
                source_type = meta.get("source_type", "property")
                if requested_source_type and source_type != requested_source_type:
                    match = False
                if filters["bhk"] is not None and meta.get("bhk") != filters["bhk"]:
                    if source_type == "property":
                        match = False
                candidate_price = meta.get("price", meta.get("price_lakhs", float("-inf")))
                if filters["min_price"] is not None and source_type in {"property", "land"} and candidate_price < filters["min_price"]:
                    match = False
                if filters["max_price"] is not None and source_type in {"property", "land"} and candidate_price > filters["max_price"]:
                    match = False
                if filters["location"] is not None:
                    loc_cand = str(meta.get("location", meta.get("area", ""))).lower()
                    if location_scope:
                        allowed_locations = {str(value).strip().lower() for value in location_scope}
                        if loc_cand not in allowed_locations:
                            match = False
                    elif filters["location"].lower() not in loc_cand:
                        match = False
                if exclude_set:
                    cand_id = str(item.get("source_id", item.get("property_id"))).strip().lower()
                    if cand_id in exclude_set or str(item.get("property_id")).lower() in exclude_set:
                        match = False
                if match:
                    filtered.append(item)

            # If filtered from FAISS pool has items, take top_k
            if filtered:
                return filtered[:top_k], filters

            # Fallback: if FAISS top pool missed exact matches, check structured dataset
            if not self.df.empty and (requested_source_type in (None, "property")):
                df_match = self.df.copy()
                if filters["bhk"] is not None:
                    df_match = df_match[df_match["bhk"] == filters["bhk"]]
                if filters["min_price"] is not None:
                    df_match = df_match[df_match["price"] >= filters["min_price"]]
                if filters["max_price"] is not None:
                    df_match = df_match[df_match["price"] <= filters["max_price"]]
                if filters["location"] is not None:
                    df_match = df_match[df_match["location"].str.lower().str.contains(filters["location"].lower(), na=False)]

                if len(df_match) > 0:
                    fallback_results = []
                    for idx, row in df_match.iterrows():
                        doc_idx = int(idx)
                        source_id = f"LEGACY-PROP-{doc_idx + 1:04d}"
                        if exclude_set and (source_id.lower() in exclude_set or str(doc_idx + 1) in exclude_set):
                            continue
                        doc_text = self.documents[doc_idx] if doc_idx < len(self.documents) else ""
                        fallback_results.append({
                            "property_id": doc_idx + 1,
                            "source_type": "property",
                            "source_id": source_id,
                            "document": doc_text,
                            "similarity": 0.5, # Nominal confidence for structured match
                            "metadata": dict(row)
                        })
                        if len(fallback_results) >= top_k:
                            break
                    return fallback_results, filters
                else:
                    # User asked for specific filters, but 0 properties exist
                    return [], filters

        # If no explicit constraints, filter candidates by semantic similarity threshold
        if exclude_set:
            candidates = [c for c in candidates if str(c.get("source_id", c.get("property_id"))).strip().lower() not in exclude_set and str(c.get("property_id")).lower() not in exclude_set]
        valid_semantic = [c for c in candidates if c["similarity"] >= min_similarity]
        return valid_semantic[:top_k], filters

    def build_prompt(self, user_question: str, context_text: str) -> str:
        """
        Constructs the strict context-grounded prompt for Gemini.
        Explicitly prevents hallucination of any external or ungrounded data.
        """
        history_str = ""
        if self.conversation_history:
            # Sliding window of last 6 turns
            recent = self.conversation_history[-6:]
            for msg in recent:
                history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"

        prompt = f"""You are an expert AI Real Estate Assistant specialized exclusively in Chennai residential properties.
Your mission is to provide accurate, reliable, and strictly grounded answers using ONLY the property information provided below.

CRITICAL HALLUCINATION PREVENTION RULES:
1. Answer the user's question using ONLY the properties and details in the "Retrieved Property Information" section.
2. NEVER INVENT OR FABRICATE ANY:
   - Property prices
   - Locations or proximity to landmarks
   - BHK configurations or dimensions
   - Builders, developer reputations, or legal approvals
   - Nearby hospitals, schools, colleges, metro stations, or distances
   - Safety ratings, quality ratings, or investment return guarantees
   - Amenities not explicitly mentioned
3. If the user asks about an ungrounded attribute (e.g. "which hospital is near", "what is the crime rate", "does it have a swimming pool", "what is the safety rating"), you MUST explicitly state:
   "Sorry, that specific information is not available in the current knowledge base."
4. If no property matches the user's requirements, you MUST explicitly state:
   "Sorry, no matching properties were found in the current knowledge base."
5. If the user asks a casual greeting or courtesy question ("Hi", "Hello", "Thanks"), respond warmly and offer to help them find Chennai homes.
6. When presenting matching properties, clearly display:
   - Property ID
   - Location
   - BHK
   - Price (in Lakhs)
   - Area (sq.ft)
   - Status (Ready To Move / Under Construction)
   - Builder
   - Bathrooms & Age
7. Use the "Previous Conversation History" solely to resolve follow-up references (e.g., "Which one is cheaper?", "Tell me more about the second one"). All property facts must still come strictly from the retrieved context.
8. If the user asks about a specific property using references like "that one", "this one", "the second one", or "the first property", select the relevant item from the retrieved results only; do not invent or assume extra properties.

Previous Conversation History:
{history_str if history_str else "No previous conversation."}

Retrieved Property Information:
{context_text}

Current User Question:
{user_question}

Grounded Response:"""
        return prompt

    def _fallback_answer(self, user_question, retrieved_items):
        if any(term in user_question.lower() for term in UNGROUNDED_ATTRIBUTE_TERMS):
            return "Sorry, that specific information is not available in the current knowledge base."

        if not retrieved_items:
            return "Sorry, no matching properties were found in the current knowledge base for your specified criteria."

        return f"I found {len(retrieved_items)} matching properties for your search. Here are the available options:"

    def _resolve_followup_targets(self, query_lower: str):
        """
        Returns a filtered subset of the last retrieved properties when the user is asking
        about a concrete prior result such as 'the second property', 'that one', or 'its area'.
        """
        if not self.last_retrieved_properties:
            return None

        q = query_lower
        last_items = self.last_retrieved_properties

        if re.search(r'\b(first|1st)\s+(?:one|property)\b', q):
            return [last_items[0]] if len(last_items) > 0 else None
        if re.search(r'\b(second|2nd)\s+(?:one|property)\b', q):
            return [last_items[1]] if len(last_items) > 1 else [last_items[-1]]
        if re.search(r'\b(third|3rd)\s+(?:one|property)\b', q):
            return [last_items[2]] if len(last_items) > 2 else [last_items[-1]]
        if re.search(r'\b(that one|this one|the last one|previous one|that property|this property)\b', q):
            return [last_items[-1]]
        if re.search(r'\b(?:what about|tell me more about)\s+(?:the\s+)?(?:first|second|third|fourth|fifth)\b', q):
            m = re.search(r'\b(first|second|third|fourth|fifth)\b', q)
            if m:
                idx_map = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
                idx = idx_map.get(m.group(1), 0)
                if len(last_items) > idx:
                    return [last_items[idx]]

        return None

    def ask(self, user_question: str, top_k=5) -> str:
        """
        End-to-end RAG inference:
        1. Checks for casual greetings
        2. Resolves broad property intent (guides without dumping)
        3. Handles 'show more' pagination
        4. Resolves property selection ('property 5', 'fifth one')
        5. Handles selected property follow-up queries (facilities, broker, price, switch property)
        6. Retrieves relevant properties from FAISS / dataset with grounding
        7. Generates grounded answer with Gemini
        8. Updates conversation history
        """
        cleaned_query = (user_question or "").strip()
        if not cleaned_query:
            return "Please enter a valid question about Chennai real estate properties."

        query_lower = cleaned_query.lower().strip("?!. ")

        if query_lower in {"new search", "clear conversation", "clear chat", "clear", "reset", "reset memory"}:
            self.clear_memory()
            return "Your search context has been cleared. How can I help you find a property today?"

        # 1. Handle simple greetings naturally
        if query_lower in GREETINGS or re.search(r"^(?:hi|hello|hey|good morning|good afternoon|good evening|namaste)\b", query_lower):
            self.last_retrieved_properties = []
            if query_lower in {"thanks", "thank you"}:
                greeting_resp = "You're welcome! Let me know whenever you'd like to explore some properties."
            elif query_lower == "how are you":
                greeting_resp = "I'm doing well! I'm ready to help you explore Chennai properties. What kind of property are you looking for?"
            elif query_lower in {"hi", "hey"}:
                greeting_resp = "Hi! I'd be happy to help you find a suitable property in Chennai. What are you looking for?"
            else:
                greeting_resp = "Hello! Welcome to Chennai Real Estate AI Assistant. How can I help you find a property today?"
            self.conversation_history.append({"role": "user", "content": cleaned_query})
            self.conversation_history.append({"role": "assistant", "content": greeting_resp})
            return greeting_resp

        # 2. Broad intent: User expressing interest in finding property without criteria
        temp_filters = extract_query_filters(query_lower)
        temp_loc = self.detect_location(query_lower)
        if is_broad_property_query(query_lower, temp_filters) and not temp_loc and not self.conversation_context.get("location"):
            self.last_retrieved_properties = []
            self.update_conversation_context(cleaned_query)
            resp = "Sure! I can help you find houses, apartments or land. You can tell me the location, budget, BHK or property type you're looking for."
            self.conversation_history.append({"role": "user", "content": cleaned_query})
            self.conversation_history.append({"role": "assistant", "content": resp})
            return resp

        # 3. Handle 'show more' pagination
        is_show_more = bool(
            re.search(r"^(?:show|show me|any)?\s*(?:more|more properties|more options|other options|additional properties|additional options)\??$", query_lower)
            or query_lower in {"show more", "show me more", "any other options", "more properties", "show more properties"}
        )
        if is_show_more:
            if not self.shown_property_ids and not self.last_retrieved_properties:
                resp = "You haven't searched for any properties yet. Tell me your preferred location, budget, or BHK to get started!"
                self.conversation_history.append({"role": "user", "content": cleaned_query})
                self.conversation_history.append({"role": "assistant", "content": resp})
                return resp

            contextual_query = self.build_contextual_query("properties")
            location_scope = self.get_search_locations() if self.conversation_context.get("nearby_requirement") else None
            more_items, _ = self.retrieve(contextual_query, top_k=top_k, location_scope=location_scope, exclude_ids=self.shown_property_ids)
            if not more_items:
                self.last_retrieved_properties = []
                resp = "I've shown all the matching properties available in the current knowledge base."
                self.conversation_history.append({"role": "user", "content": cleaned_query})
                self.conversation_history.append({"role": "assistant", "content": resp})
                return resp

            self.last_retrieved_properties = more_items
            self.shown_property_ids.update(str(item.get("source_id", item.get("property_id"))) for item in more_items)
            resp = f"Here are {len(more_items)} additional matching properties available in the knowledge base:"
            self.conversation_history.append({"role": "user", "content": cleaned_query})
            self.conversation_history.append({"role": "assistant", "content": resp})
            return resp

        # 4. Handle direct property selection (e.g., 'Tell me about property 5', 'I want to see property 5', 'What about the fifth one?')
        selection_target, target_idx = extract_property_selection(query_lower, self.last_retrieved_properties)
        if selection_target is not None:
            from property_details import get_property_record
            record = get_property_record(selection_target)
            self.set_selected_property(record)

            bhk = f"{record.get('bhk')} BHK " if record.get('bhk') and not pd.isna(record.get('bhk')) else ""
            p_type = record.get('property_type', record.get('land_type', 'Property'))
            loc = record.get('location', 'Chennai')
            price = record.get('price_lakhs', record.get('price', '—'))
            p_id = record.get('property_id', selection_target.get('source_id', ''))

            resp = (
                f"Here are the details for Property {target_idx + 1} ({p_id}):\n\n"
                f"🏠 **{bhk}{p_type} in {loc}**\n"
                f"💰 **Price:** ₹{price} Lakhs | 📐 **Area:** {record.get('area_sqft', record.get('area', '—'))} sq.ft\n"
                f"📍 **Address:** {record.get('street', '')}, {loc}, {record.get('city', 'Chennai')}\n\n"
                f"{record.get('description', '')}\n\n"
                f"You can view its photos, specifications, and nearby facilities below. "
                f"Feel free to ask questions like 'What schools are nearby?', 'What about hospitals?', or 'Who is the broker?'"
            )
            self.conversation_history.append({"role": "user", "content": cleaned_query})
            self.conversation_history.append({"role": "assistant", "content": resp})
            return resp

        # 5. Selected property follow-ups
        if self.selected_property_context:
            # A. 'Show me another property'
            if re.search(r"\b(show me another property|another property|back to results|back to properties|show another|other properties)\b", query_lower):
                self.clear_selected_property()
                resp = "Returning to your property search results. You can select another property to view its details or refine your criteria."
                self.conversation_history.append({"role": "user", "content": cleaned_query})
                self.conversation_history.append({"role": "assistant", "content": resp})
                return resp

            # B. Broker inquiry for selected property
            if re.search(r"\b(who is the broker|who to contact|who can i contact|broker|contact details|contact person|contact info|phone number|broker info|broker's name)\b", query_lower):
                from property_details import get_broker
                broker_id = self.selected_property_context.get("broker_id")
                broker = get_broker(broker_id)
                if broker:
                    resp = (
                        f"For Property {self.selected_property_context.get('property_id', '')}, here is the demo broker contact:\n\n"
                        f"- **Broker:** {broker.get('broker_name', 'Demo Broker')}\n"
                        f"- **Agency:** {broker.get('agency_name', 'Real Estate Agency')}\n"
                        f"- **Phone:** {broker.get('phone', 'Not specified')}\n"
                        f"- **Email:** {broker.get('email', 'Not specified')}\n"
                        f"- **Areas Served:** {broker.get('areas_served', 'Chennai')}"
                    )
                else:
                    resp = "Demo broker contact is not available for this listing."
                self.conversation_history.append({"role": "user", "content": cleaned_query})
                self.conversation_history.append({"role": "assistant", "content": resp})
                return resp

            # C. Price inquiry for selected property
            if re.search(r"\b(how much is this|what is the price|price of this|cost of this|how much does it cost|what's the price)\b", query_lower):
                price = self.selected_property_context.get("price_lakhs", self.selected_property_context.get("price", "—"))
                resp = f"The selected property ({self.selected_property_context.get('property_id', '')}) is priced at ₹{price} Lakhs."
                self.conversation_history.append({"role": "user", "content": cleaned_query})
                self.conversation_history.append({"role": "assistant", "content": resp})
                return resp

            # D. Facility inquiries for selected property (Schools, Hospitals, etc.)
            if any(k in query_lower for k in ["school", "hospital", "college", "supermarket", "transport", "bus stop", "metro", "railway"]):
                from property_details import get_nearby_facilities_for_property
                facilities = get_nearby_facilities_for_property(self.selected_property_context)
                if "school" in query_lower:
                    fac_type, label = "school", "schools"
                elif "hospital" in query_lower:
                    fac_type, label = "hospital", "hospitals"
                elif "college" in query_lower:
                    fac_type, label = "college", "colleges"
                elif "supermarket" in query_lower or "market" in query_lower:
                    fac_type, label = "supermarket", "supermarkets"
                else:
                    fac_type, label = "transport", "transit stations"

                rows = facilities.get(fac_type, [])
                if rows:
                    lines = []
                    for r in rows:
                        name = r.get(f"{fac_type}_name", r.get("transport_name", "Facility"))
                        area = r.get("area", "Chennai")
                        dist = r.get("distance_km", 0.0)
                        lines.append(f"- **{name}** ({area}) — {dist:.1f} km")
                    resp = f"Here are the nearest {label} to the selected property ({self.selected_property_context.get('property_id', '')}):\n\n" + "\n".join(lines)
                else:
                    resp = f"No nearby {label} are available for this property in the current knowledge base."
                self.conversation_history.append({"role": "user", "content": cleaned_query})
                self.conversation_history.append({"role": "assistant", "content": resp})
                return resp

        # 6. Standard RAG Retrieval
        followup_targets = self._resolve_followup_targets(query_lower)
        is_followup = bool(
            (
                re.search(r'\b(which one|cheaper|lowest|first one|second one|third one|compare|previous|above|what about the|tell me more about|that one|this one|show me cheaper options|what is its area|what is the area)\b', query_lower)
                or (followup_targets is not None)
            )
            and self.last_retrieved_properties
        )

        if is_followup:
            retrieved_items = followup_targets or self.last_retrieved_properties
        else:
            contextual_query = self.build_contextual_query(cleaned_query)
            if self.needs_location_for_nearby():
                response = "Which Chennai area should I search around?"
                self.conversation_history.append({"role": "user", "content": cleaned_query})
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
            location_scope = self.get_search_locations() if self.conversation_context.get("nearby_requirement") else None
            retrieved_items, filters = self.retrieve(contextual_query, top_k=top_k, location_scope=location_scope)
            self.last_retrieved_properties = retrieved_items
            if retrieved_items:
                self.shown_property_ids = {str(item.get("source_id", item.get("property_id"))) for item in retrieved_items}

        # Guardrail: If no matching properties found
        if not retrieved_items:
            refusal = "Sorry, no matching properties were found in the current knowledge base for your specified criteria."
            self.conversation_history.append({"role": "user", "content": cleaned_query})
            self.conversation_history.append({"role": "assistant", "content": refusal})
            return refusal

        # Format context for Gemini
        context_parts = []
        for i, item in enumerate(retrieved_items, 1):
            context_parts.append(f"[Candidate Property {i}]\n{item['document']}")

        context_text = "\n\n".join(context_parts)

        # Build prompt & query Gemini with safe retry and strictly guarded fallback
        prompt = self.build_prompt(cleaned_query, context_text)
        answer = ""
        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                response_text = getattr(response, "text", None)
                if not response_text or not response_text.strip():
                    raise ValueError("Gemini returned an empty response")
                answer = response_text.strip()
                break
            except Exception as e:
                err_str = str(e)
                transient_error = any(code in err_str.lower() for code in ["503", "429", "unavailable", "resourceexhausted", "rate limit", "overloaded", "timed out"])
                if attempt < max_retries - 1 and transient_error:
                    import time
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                # Professional fallback without exposing internal error tracebacks
                answer = self._fallback_answer(cleaned_query, retrieved_items)
                break

        self.conversation_history.append({"role": "user", "content": cleaned_query})
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def _resolve_property_record(self, result):
        from property_details import get_property_record, load_property_catalog
        return get_property_record(result, load_property_catalog())

    def _property_followup_response(self, query: str):
        if not self.last_retrieved_properties:
            return None

        query_lower = (query or "").lower().strip("?!. ")
        if re.search(r"\b(which one is cheaper|which is the cheapest|cheaper|cheapest|more expensive|most expensive)\b", query_lower):
            candidates = []
            for item in self.last_retrieved_properties:
                record = self._resolve_property_record(item)
                price = record.get("price_lakhs", record.get("price"))
                if price is not None and not pd.isna(price):
                    candidates.append((float(price), record))
            if not candidates:
                return None
            if "more expensive" in query_lower or "most expensive" in query_lower or "expensive" in query_lower:
                selected = max(candidates, key=lambda x: x[0])[1]
            else:
                selected = min(candidates, key=lambda x: x[0])[1]
            self.selected_property_context = dict(selected)
            price_text = selected.get("price_lakhs", selected.get("price", "—"))
            return (
                f"The {'most expensive' if 'expensive' in query_lower or 'most expensive' in query_lower else 'cheapest'} property in the current results is "
                f"{selected.get('property_id', 'this property')} in {selected.get('location', 'Chennai')} at ₹{price_text} Lakhs."
            )

        if self.selected_property_context is None:
            self.selected_property_context = self._resolve_property_record(self.last_retrieved_properties[-1])

        selected = self.selected_property_context
        if not selected:
            return None

        if re.search(r"\b(what is the (area|square feet|sq\.?ft)|what is the area|what is the square feet of this property|what is the square footage)\b", query_lower):
            area = selected.get("area_sqft", selected.get("area", "—"))
            return f"The selected property ({selected.get('property_id', '')}) has an area of {area} sq.ft."

        if re.search(r"\b(how many bathrooms|bathrooms|bathroom)\b", query_lower):
            baths = selected.get("bathrooms", "—")
            return f"The selected property ({selected.get('property_id', '')}) has {baths} bathrooms."

        if re.search(r"\b(where is this|what is the location|where is it)\b", query_lower):
            loc = selected.get("location", "Chennai")
            street = selected.get("street", "")
            return f"The selected property is in {loc}{f', {street}' if street else ''}."

        if re.search(r"\b(what is the price|what is the cost|how much is this|price of this|how much does it cost)\b", query_lower):
            price = selected.get("price_lakhs", selected.get("price", "—"))
            return f"The selected property ({selected.get('property_id', '')}) is priced at ₹{price} Lakhs."

        if re.search(r"\b(tell me about this property|describe this property|about this property)\b", query_lower):
            return (
                f"Selected property {selected.get('property_id', '')}: {selected.get('property_type', 'Property')} in {selected.get('location', 'Chennai')} with "
                f"{selected.get('bhk', '—')} BHK, {selected.get('area_sqft', selected.get('area', '—'))} sq.ft, and price ₹{selected.get('price_lakhs', selected.get('price', '—'))} Lakhs."
            )

        if re.search(r"\b(what schools are nearby|schools nearby|school nearby|colleges nearby|hospitals nearby|supermarkets nearby|nearby schools|nearby colleges|nearby hospitals|nearby supermarkets)\b", query_lower):
            from property_details import get_nearby_facilities_for_property
            facilities = get_nearby_facilities_for_property(selected)
            facility_type = "school" if "school" in query_lower else "college" if "college" in query_lower else "hospital" if "hospital" in query_lower else "supermarket" if "supermarket" in query_lower else "transport"
            rows = facilities.get(facility_type, [])
            if not rows:
                return f"No nearby {facility_type} information is available in the current knowledge base for the selected property."
            lines = []
            for row in rows[:3]:
                name = row.get(f"{facility_type}_name", row.get("transport_name", "Facility"))
                lines.append(f"- {name} ({row.get('area', 'Chennai')}) — {row.get('distance_km', 0.0):.1f} km")
            return f"Nearby {facility_type} for the selected property are:\n" + "\n".join(lines)

        return None

    def clear_memory(self):
        """
        Clears conversation history and cached properties.
        """
        self.conversation_history.clear()
        self.last_retrieved_properties.clear()
        self.shown_property_ids.clear()
        self.clear_selected_property()
        for key in self.conversation_context:
            self.conversation_context[key] = [] if key == "nearby_locations" else None

def main():
    print("=" * 65)
    print("  🏠 CHENNAI REAL ESTATE AI CHATBOT (RAG + FAISS + GEMINI)")
    print("=" * 65)
    print("Ask natural-language questions about Chennai real estate.")
    print("Examples:")
    print("  - Find 3 BHK houses in Anna Nagar")
    print("  - Show properties under 50 lakhs")
    print("  - Do you have ready to move apartments in Chennai?")
    print("  - Which one is cheaper? (Follow-up)")
    print("Type 'clear' to reset memory, or 'exit' to quit.\n")

    try:
        bot = RealEstateRAG()
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Thank you for using Chennai Real Estate Assistant. Goodbye!")
            break

        if user_input.lower() == "clear":
            bot.clear_memory()
            print("[System] Conversation history and memory cleared.\n")
            continue

        response = bot.ask(user_input)
        print("\nAssistant:")
        print(response)
        print("\n" + "-" * 65 + "\n")

if __name__ == "__main__":
    main()