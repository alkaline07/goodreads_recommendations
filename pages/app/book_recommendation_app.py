import streamlit as st
import streamlit.components.v1 as components
import requests
from typing import List, Dict
import json
import os
import time
from datetime import datetime
from urllib.parse import quote

# Restore session state BEFORE Streamlit renders widgets
current_user = st.session_state.get("current_user", None)

API_BASE_URL = "https://recommendation-service-491512947755.us-central1.run.app"

# [Keep all your existing helper functions exactly as they are - inject_web_vitals_monitoring, format_rating, 
# track_api_call, get_recommendations, get_read_books, get_unread_books, load_books_database, 
# load_fallback_database, search_books, get_book_details_from_google, create_fallback_book_details, 
# send_click_event - NO CHANGES TO THESE]

def inject_web_vitals_monitoring(session_id: str, api_base_url: str):
    """Inject Web Vitals monitoring script into the Streamlit app's parent window."""
    web_vitals_script = f"""
    <script type="module">
        const sessionId = '{session_id}';
        const apiBaseUrl = '{api_base_url}';
        
        // Check if already initialized in parent window
        const parentWin = window.parent || window;
        if (parentWin._webVitalsInitialized) {{
            console.log('[Web Vitals] Already initialized');
        }} else {{
            parentWin._webVitalsInitialized = true;
            parentWin._webVitals = {{}};
            parentWin._lastVitalsSend = 0;
            parentWin._clsValue = 0;
            parentWin._inpEntries = [];
            
            function sendVitals(force = false) {{
                const now = Date.now();
                if (!force && now - parentWin._lastVitalsSend < 5000) return;
                
                const vitals = parentWin._webVitals || {{}};
                if (Object.keys(vitals).length === 0) return;
                
                parentWin._lastVitalsSend = now;
                
                const payload = {{
                    sessionId: sessionId,
                    page: parentWin.location.pathname || '/streamlit',
                    userAgent: navigator.userAgent,
                    timestamp: new Date().toISOString(),
                    metrics: {{
                        webVitals: {{...vitals}},
                        apiCalls: [],
                        errors: []
                    }}
                }};
                
                fetch(apiBaseUrl + '/frontend-metrics', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(payload),
                    keepalive: true
                }}).then(() => console.log('[Web Vitals] Sent:', vitals))
                  .catch(e => console.warn('[Web Vitals] Failed to send:', e));
            }}
            
            function calculateINP() {{
                const entries = parentWin._inpEntries;
                if (entries.length === 0) return;
                entries.sort((a, b) => b - a);
                const p98Index = Math.min(Math.floor(entries.length * 0.02), entries.length - 1);
                parentWin._webVitals.inp = Math.round(entries[p98Index]);
            }}
            
            try {{
                new PerformanceObserver((list) => {{
                    const entries = list.getEntries();
                    const lastEntry = entries[entries.length - 1];
                    parentWin._webVitals.lcp = Math.round(lastEntry.startTime);
                    sendVitals();
                }}).observe({{type: 'largest-contentful-paint', buffered: true}});
                
                new PerformanceObserver((list) => {{
                    for (const entry of list.getEntries()) {{
                        if (!entry.hadRecentInput) {{
                            parentWin._clsValue += entry.value;
                        }}
                    }}
                    parentWin._webVitals.cls = parseFloat(parentWin._clsValue.toFixed(4));
                    sendVitals();
                }}).observe({{type: 'layout-shift', buffered: true}});
                
                new PerformanceObserver((list) => {{
                    const entries = list.getEntries();
                    if (entries.length > 0) {{
                        parentWin._webVitals.fcp = Math.round(entries[0].startTime);
                        sendVitals();
                    }}
                }}).observe({{type: 'paint', buffered: true}});
                
                new PerformanceObserver((list) => {{
                    for (const entry of list.getEntries()) {{
                        if (entry.duration > 0) {{
                            parentWin._inpEntries.push(entry.duration);
                            calculateINP();
                        }}
                    }}
                    sendVitals();
                }}).observe({{type: 'event', buffered: true, durationThreshold: 0}});
                
                const navEntry = performance.getEntriesByType('navigation')[0];
                if (navEntry) {{
                    parentWin._webVitals.ttfb = Math.round(navEntry.responseStart);
                }}
                
                console.log('[Web Vitals] Monitoring initialized for session:', sessionId);
                
            }} catch (e) {{
                console.warn('[Web Vitals] PerformanceObserver not fully supported:', e);
            }}
            
            function trackInteraction(event) {{
                const start = performance.now();
                requestAnimationFrame(() => {{
                    requestAnimationFrame(() => {{
                        const duration = performance.now() - start;
                        if (duration > 0) {{
                            parentWin._inpEntries.push(duration);
                            calculateINP();
                            sendVitals();
                        }}
                    }});
                }});
            }}
            
            ['click', 'keydown', 'pointerdown'].forEach(eventType => {{
                parentWin.document.addEventListener(eventType, trackInteraction, {{passive: true, capture: true}});
            }});
            
            try {{
                document.addEventListener('click', trackInteraction, {{passive: true, capture: true}});
                document.addEventListener('keydown', trackInteraction, {{passive: true, capture: true}});
            }} catch (e) {{}}
            
            const mutationObserver = new MutationObserver(() => {{
                if (parentWin._webVitals.cls === undefined) {{
                    parentWin._webVitals.cls = 0;
                }}
            }});
            mutationObserver.observe(parentWin.document.body, {{
                childList: true,
                subtree: true,
                attributes: true
            }});
            
            setTimeout(sendVitals, 3000);
            setInterval(sendVitals, 30000);
            
            parentWin.addEventListener('visibilitychange', () => {{
                if (parentWin.document.visibilityState === 'hidden') {{
                    calculateINP();
                    sendVitals(true);
                }}
            }});
            parentWin.addEventListener('pagehide', () => {{ calculateINP(); sendVitals(true); }});
            parentWin.addEventListener('beforeunload', () => {{ calculateINP(); sendVitals(true); }});
        }}
    </script>
    """
    components.html(web_vitals_script, height=0, width=0)

def format_rating(rating):
    """Safely format rating for display"""
    if rating is None or rating == "N/A":
        return "N/A"
    try:
        return f"{float(rating):.1f}"
    except (ValueError, TypeError):
        return "N/A"

def track_api_call(endpoint: str, method: str, duration_ms: float, status_code: int, error: str = None):
    """Track API call metrics to backend monitoring"""
    try:
        if 'api_call_count' not in st.session_state:
            st.session_state.api_call_count = 0
        if 'total_api_latency' not in st.session_state:
            st.session_state.total_api_latency = 0.0

        st.session_state.api_call_count += 1
        st.session_state.total_api_latency += duration_ms

        metrics_payload = {
            "sessionId": st.session_state.get('session_id', 'unknown'),
            "timestamp": datetime.utcnow().isoformat(),
            "url": endpoint,
            "metrics": {
                "apiCalls": [{
                    "endpoint": endpoint,
                    "method": method,
                    "latency": duration_ms,
                    "status": "success" if status_code == 200 else "error",
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
        }

        requests.post(
            f"{API_BASE_URL}/frontend-metrics",
            json=metrics_payload,
            timeout=2
        )
    except:
        pass


def get_recommendations(user_id: str) -> List[Dict]:
    """Call FastAPI /load-recommendation"""
    start_time = time.time()
    try:
        url = f"{API_BASE_URL}/load-recommendation"
        resp = requests.post(url, json={"user_id": user_id}, timeout=15)
        duration_ms = (time.time() - start_time) * 1000

        track_api_call("/load-recommendation", "POST", duration_ms, resp.status_code)

        if resp.status_code == 200:
            return resp.json().get("recommendations", [])
        else:
            st.error(f"API Error: {resp.status_code}")
            return []
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        track_api_call("/load-recommendation", "POST", duration_ms, 500, str(e))
        st.error(f"Error fetching recommendations: {e}")
        return []


def get_read_books(user_id: str) -> List[Dict]:
    """GET /books-read/{user_id}"""
    start_time = time.time()
    try:
        url = f"{API_BASE_URL}/books-read/{user_id}"
        resp = requests.get(url, timeout=10)
        duration_ms = (time.time() - start_time) * 1000

        track_api_call(f"/books-read/{user_id}", "GET", duration_ms, resp.status_code)

        if resp.status_code == 200:
            return resp.json().get("books_read", [])
        return []
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        track_api_call(f"/books-read/{user_id}", "GET", duration_ms, 500, str(e))
        st.error(f"Error fetching read books: {e}")
        return []

def get_unread_books(user_id: str) -> List[Dict]:
    """GET /books-unread/{user_id}"""
    start_time = time.time()
    try:
        url = f"{API_BASE_URL}/books-unread/{user_id}"
        resp = requests.get(url, timeout=10)
        duration_ms = (time.time() - start_time) * 1000

        track_api_call(f"/books-unread/{user_id}", "GET", duration_ms, resp.status_code)

        if resp.status_code == 200:
            return resp.json().get("books_unread", [])
        return []
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        track_api_call(f"/books-unread/{user_id}", "GET", duration_ms, 500, str(e))
        st.error(f"Error fetching unread books: {e}")
        return []

def load_books_database():
    """Load books database from JSON file"""
    try:
        json_path = 'data/books_database.json'
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

                books = []
                for item in raw_data:
                    books.append({
                        "book_id": str(item.get("book_id", "")),
                        "title": item.get("book_title") or item.get("title_clean") or item.get("title") or "Unknown Title",
                        "author": item.get("author_name") or item.get("authors_flat") or "Unknown Author",
                        "rating": float(item.get("average_rating") or 3.5),
                        "isbn": item.get("isbn") or item.get("isbn_clean") or ""
                    })

                return books
        else:
            st.warning("books_database.json not found, using fallback data")
            return load_fallback_database()
    except Exception as e:
        st.error(f"Error loading books database: {str(e)}")
        return load_fallback_database()


def load_fallback_database():
    """Fallback database if JSON file is not available"""
    return [
        {"book_id": "13079104", "title": "Circe", "author": "Madeline Miller", "rating": 4.6, "isbn": "9780316556347"},
        {"book_id": "11295686", "title": "Gone Girl", "author": "Gillian Flynn", "rating": 4.4, "isbn": "9780307588371"},
    ]


def search_books(query: str, books_db: List[Dict]) -> List[Dict]:
    """Search books by title or author"""
    if not query:
        return books_db[:10]

    query_lower = query.lower()
    results = [
        book for book in books_db
        if query_lower in book.get('title', '').lower() or query_lower in book.get('author', '').lower()
    ]
    return results[:20]


def get_book_details_from_google(title: str, author: str, isbn: str = None, api_rating=None) -> Dict:
    """Fetch book details from Google Books API with improved search accuracy"""
    try:
        book_data = None

        if isbn:
            url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    book_data = data['items'][0]['volumeInfo']

        if not book_data:
            encoded_title = quote(title)
            encoded_author = quote(author)
            url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{encoded_title}+inauthor:{encoded_author}"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    best_match = None
                    for item in data['items']:
                        item_info = item.get('volumeInfo', {})
                        item_title = item_info.get('title', '').lower()
                        item_authors = [a.lower() for a in item_info.get('authors', [])]

                        if title.lower() == item_title:
                            if any(author.lower() in a or a in author.lower() for a in item_authors):
                                best_match = item_info
                                break

                        if best_match is None:
                            best_match = item_info

                    book_data = best_match

        if not book_data:
            query = f"{title} {author}".replace(" ", "+")
            url = f"https://www.googleapis.com/books/v1/volumes?q={quote(query, safe='+')}"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    book_data = data['items'][0]['volumeInfo']

        if book_data:
            if api_rating is not None and api_rating != "N/A":
                final_rating = api_rating
            else:
                final_rating = book_data.get('averageRating', 'N/A')

            return {
                "title": book_data.get('title', title),
                "author": ', '.join(book_data.get('authors', [author])),
                "description": book_data.get('description', 'No description available.'),
                "cover_url": book_data.get('imageLinks', {}).get('thumbnail',
                                                                 book_data.get('imageLinks', {}).get(
                                                                     'smallThumbnail',
                                                                     'https://via.placeholder.com/300x450?text=No+Cover')),
                "published_date": book_data.get('publishedDate', 'Unknown'),
                "page_count": book_data.get('pageCount', 'N/A'),
                "categories": book_data.get('categories', ['General']),
                "rating": final_rating,
                "ratings_count": book_data.get('ratingsCount', 'N/A'),
                "language": book_data.get('language', 'en'),
                "preview_link": book_data.get('previewLink', '#')
            }

        return create_fallback_book_details(title, author, api_rating)

    except Exception as e:
        st.error(f"Error fetching book details: {str(e)}")
        return create_fallback_book_details(title, author, api_rating)


def create_fallback_book_details(title: str, author: str, api_rating=None) -> Dict:
    """Create fallback book details if Google Books API fails"""
    return {
        "title": title,
        "author": author,
        "description": "Description not available. Please try again later.",
        "cover_url": "https://via.placeholder.com/300x450?text=No+Cover+Available",
        "published_date": "Unknown",
        "page_count": "N/A",
        "categories": ["General"],
        "rating": api_rating if api_rating is not None else "N/A",
        "ratings_count": "N/A",
        "language": "en",
        "preview_link": "#"
    }



def send_click_event(user_id: str, book_id: int, event_type: str, book_title: str = None):
    """function to send click/interaction events for CTR tracking"""
    event_data = {
        "user_id": user_id,
        "book_id": book_id,
        "book_title": book_title,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        response = requests.post(f"{API_BASE_URL}/book-click", json=event_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error tracking event: {e}")
        return {"status": "error", "message": str(e)}


# Initialize session state
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'read_books' not in st.session_state:
    st.session_state.read_books = []
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'user_selection'
if 'books_database' not in st.session_state:
    st.session_state.books_database = load_books_database()
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'show_admin_modal' not in st.session_state:
    st.session_state.show_admin_modal = False
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if 'api_call_count' not in st.session_state:
    st.session_state.api_call_count = 0
if 'total_api_latency' not in st.session_state:
    st.session_state.total_api_latency = 0.0
if 'previous_view' not in st.session_state:
    st.session_state.previous_view = 'recommendations'
if 'last_search_query' not in st.session_state:
    st.session_state.last_search_query = ""
if 'vitals_injected' not in st.session_state:
    st.session_state.vitals_injected = False
if 'show_rating_modal' not in st.session_state:
    st.session_state.show_rating_modal = False
if 'rating_book_info' not in st.session_state:
    st.session_state.rating_book_info = None

# Page configuration
st.set_page_config(
    page_title="Read Mate",
    page_icon="📚",
    layout="wide"
)

# ===== UPDATED MODERN CSS =====
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    .book-card {
        padding: 24px;
        border-radius: 16px;
        border: none;
        margin: 16px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
        min-height: 180px;
        display: flex; flex-direction: column; justify-content: space-between;
    }
    
    .book-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
        
    }
    
    .book-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.12), 0 8px 16px rgba(0,0,0,0.08);
    }
    
    .book-card:hover::before {
        opacity: 1;
    }
    
    .book-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #1a202c;
        line-height: 1.4;
        letter-spacing: -0.01em;
        min-height: 56px;
        -webkit-line-clamp: 2;
        overflow: hidden;
    }
    
    .book-author {
        color: #718096;
        margin-bottom: 12px;
        font-size: 15px;
        font-weight: 400;
    }
    
    .rating {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
    }
    
    .detail-section {
        margin: 32px 0;
        padding: 24px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .genre-tag {
        display: inline-block;
        padding: 8px 16px;
        margin: 6px 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 24px;
        font-size: 13px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        transition: all 0.2s ease;
    }
    
    .genre-tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .description-text {
        line-height: 1.8;
        text-align: justify;
        color: #2d3748;
        font-size: 16px;
        padding: 20px;
        background: #f7fafc;
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    .stButton > button[kind="secondary"] {
    background: white;
    color: #1a202c;
    border: 2px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .stButton > button[kind="secondary"]:hover {
        background: #f7fafc;
        border-color: #cbd5e0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        letter-spacing: 0.02em;
        height: 40px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar horizontal rules */
    [data-testid="stSidebar"] hr {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.2);
        margin: 16px 0;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .stSuccess, .stInfo {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 16px;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 12px 16px;
        font-size: 15px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSpinner > div {
        border-color: #667eea;
    }
    
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 32px 0;
    }
/* Action buttons styling */
    .action-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 14px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        border: none;
        width: 100%;
    }

    .like-button {
        background: linear-gradient(135deg, #ff6b9d 0%, #c06c84 100%);
        /* Pink gradient for Like button */
    }

    .reading-list-button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        /* Blue gradient for Reading List */
    }

    .mark-read-button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        /* Green gradient for Mark as Read */
    }
    .stButton > button[kind="primary"] {
        background: rgba(138, 43, 226);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);

    }   
     
    </style>
""", unsafe_allow_html=True)

# Inject Web Vitals monitoring
inject_web_vitals_monitoring(st.session_state.session_id, API_BASE_URL)

# Sidebar
with st.sidebar:
    st.title("Navigation")

    st.markdown("---")

    if st.session_state.current_user:
        st.markdown(f"**User:** User-{st.session_state.current_user[:4]}")
        st.markdown("---")

        if st.button("Home", use_container_width=True, key="nav_home"):
            if current_user:
                st.session_state.recommendations = get_recommendations(current_user)
                for book in st.session_state.recommendations:
                    send_click_event(current_user, book["book_id"], "view", book["title"])
            st.session_state.view_mode = "recommendations"
            st.session_state.show_rating_modal = False
            st.rerun()

        if st.button("My Read Books", use_container_width=True, key="nav_read"):
            st.session_state.read_books = get_read_books(st.session_state.current_user)
            st.session_state.view_mode = 'read_books'
            st.session_state.show_rating_modal = False
            st.rerun()

        if st.button("Search Books", use_container_width=True, key="nav_search"):
            st.session_state.view_mode = 'search'
            st.session_state.show_rating_modal = False
            st.rerun()

        st.markdown("---")

        if st.button("Logout", use_container_width=True, key="nav_logout"):
            st.session_state.current_user = None
            st.session_state.view_mode = 'user_selection'
            st.session_state.show_rating_modal = False
            st.rerun()
    else:
        st.info("Please login to access all features")


# ===== UPDATED DISPLAY BOOK CARD FUNCTION =====
def display_book_card(book: Dict, col, button_prefix: str = "btn"):
    with col:
        rating_value = format_rating(book.get('predicted_rating'))
        rating_color = "#ffd700" if rating_value != "N/A" else "#cbd5e0"
        
        st.markdown(f"""
            <div class="book-card">
                <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;'>
                    <div style='flex: 1;'>
                        <div class="book-title">{book['title']}</div>
                        <div class="book-author"><i>by {book['author']}</i></div>
                    </div>
                    <div style='background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); 
                                padding: 8px 12px; border-radius: 20px; 
                                box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
                                white-space: nowrap; margin-left: 12px;'>
                        <span style='color: white; font-weight: 600; font-size: 14px;'>⭐ {rating_value}</span>
                    </div>
                </div>
                
            </div>
        """, unsafe_allow_html=True)

        if st.button(f"View Details", key=f"{button_prefix}_{book['book_id']}", use_container_width=True):
            send_click_event(st.session_state.current_user, book['book_id'], "click", book['title'])

            with st.spinner(f'✨ Loading details for "{book["title"]}"...'):
                api_rating = book.get('predicted_rating') or book.get('average_rating') or book.get('rating')
                book_details = get_book_details_from_google(
                    book['title'],
                    book['author'],
                    book.get('isbn'),
                    api_rating=api_rating
                )
                book_details['book_id'] = book['book_id']
                st.session_state.selected_book = book_details
                st.session_state.previous_view = 'recommendations'
                st.session_state.view_mode = 'book_details'
            st.rerun()


# ===== MAIN APPLICATION VIEWS =====

# USER SELECTION VIEW (Updated with Hero Section)
if st.session_state.view_mode == 'user_selection':
    col_empty, col_admin = st.columns([11, 1])
    with col_admin:
        if st.button("🔐", use_container_width=True, help="Admin Login"):
            st.session_state.show_admin_modal = True
            st.rerun()
    if st.session_state.show_admin_modal:
        with st.container():
            
            st.markdown("### Admin Login")
            admin_username = st.text_input("Username:", key="admin_user")
            admin_password = st.text_input("Password:", type="password", key="admin_pass")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", type="primary", use_container_width=True):
                    if admin_username == "admin" and admin_password == "admin":
                        st.success("Login successful! Redirecting to monitoring dashboard...")
                        admin_url = f"{API_BASE_URL}/report"
                        st.markdown(f'<meta http-equiv="refresh" content="1;url={admin_url}" />', unsafe_allow_html=True)
                        st.markdown(f"[Click here if not redirected]({admin_url})")
                    else:
                        st.error("Invalid credentials!")
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_admin_modal = False
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    st.markdown("""
        <div style='text-align: center; padding: 60px 20px 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 24px; margin-bottom: 40px; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);'>
            <h1 style='color: white; font-size: 48px; font-weight: 700; margin-bottom: 16px; text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                 Discover Your Next Favorite Book
            </h1>
            <p style='color: rgba(255,255,255,0.95); font-size: 20px; max-width: 600px; margin: 0 auto; line-height: 1.6;'>
                Powered by advanced machine learning to recommend books tailored just for you
            </p>
        </div>
    """, unsafe_allow_html=True)

    
    st.markdown("### Get Started")
    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_input("", placeholder="Enter your User ID (e.g., user_12345)", label_visibility="collapsed")
    with col2:
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            if user_input:
                st.session_state.current_user = user_input
                st.session_state.recommendations = get_recommendations(user_input)
                st.session_state.view_mode = 'recommendations'
                for book in st.session_state.recommendations:
                    send_click_event(user_input, book['book_id'], "view", book['title'])
                st.rerun()
            else:
                st.error("Please enter a user ID")
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Try Sample Users")
    
    quick_users = {
        "User-4b1af": {"id": "4b1af908229844ec02bc4b40aa6ea4dd"},
        "User-2faa2": {"id": "2faa2ef7e9062a7339ed1e4299c7ecaf"},
        "User-4597b": {"id": "4597ba0bb52054eae1e87534c78b13b8"}
    }
    
    cols = st.columns(3)
    for idx, (display_name, user_data) in enumerate(quick_users.items()):
        with cols[idx]:
            
            if st.button(display_name, key=f"user_{idx}",use_container_width=True,type="secondary"):
                with st.spinner('Loading recommendations...'): 
                    st.session_state.current_user = user_data['id']
                    st.session_state.recommendations = get_recommendations(user_data['id'])
                    for book in st.session_state.recommendations:
                        send_click_event(user_data['id'], book['book_id'], "view", book['title'])
                    st.session_state.view_mode = 'recommendations'  
                st.rerun()

    st.markdown("---")
    st.markdown("### Features")
    
    feature_cols = st.columns(4)
    features = [
        {"icon": "🎯", "title": "Personalized", "desc": "ML-powered recommendations"},
        {"icon": "📊", "title": "Smart Analytics", "desc": "Track your reading habits"},
        {"icon": "🔍", "title": "Advanced Search", "desc": "Find any book instantly"},
        {"icon": "⚡", "title": "Real-time", "desc": "Live performance metrics"}
    ]
    
    for col, feature in zip(feature_cols, features):
        with col:
            st.markdown(f"""
                <div style='text-align: center; padding: 20px;'>
                    <div style='font-size: 32px; margin-bottom: 8px;'>{feature['icon']}</div>
                    <div style='font-weight: 600; font-size: 14px; color: #1a202c; margin-bottom: 4px;'>{feature['title']}</div>
                    <div style='font-size: 12px; color: #718096;'>{feature['desc']}</div>
                </div>
            """, unsafe_allow_html=True)

# RECOMMENDATIONS VIEW (Updated)
elif st.session_state.view_mode == 'recommendations':
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
            <div style='margin-bottom: 8px;'>
                <h1 style='margin: 0; font-size: 36px; color: #1a202c;'>Your Recommendations</h1>
                <p style='color: #718096; font-size: 16px; margin-top: 8px;'>
                    User-{st.session_state.current_user[:4]} • {len(st.session_state.recommendations)} personalized picks
                </p>
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.recommendations:
        for i in range(0, len(st.session_state.recommendations), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(st.session_state.recommendations):
                    book = st.session_state.recommendations[i + j]
                    display_book_card(book, cols[j], "rec")
    else:
        st.markdown("""
            <div style='text-align: center; padding: 60px 20px; background: white; border-radius: 16px;'>
                <div style='font-size: 64px; margin-bottom: 16px;'>📚</div>
                <h3 style='color: #1a202c; margin-bottom: 8px;'>No recommendations yet</h3>
                <p style='color: #718096;'>Start by rating some books to get personalized recommendations!</p>
            </div>
        """, unsafe_allow_html=True)

# READ BOOKS VIEW (Keep your existing code)
elif st.session_state.view_mode == 'read_books':
    st.title(f"My Read Books")
    st.markdown(f"### Books you've already read")

    if st.session_state.read_books:
        for i in range(0, len(st.session_state.read_books), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(st.session_state.read_books):
                    book = st.session_state.read_books[i + j]
                    with cols[j]:
                        st.markdown(f"""
                            <div class="book-card">
                                <div class="book-title">{book['title']}</div>
                                <div class="book-author">by {book['author']}</div>
                                <div class="rating">⭐ {format_rating(book.get('average_rating'))}</div>
                                <div style="color: #27ae60; font-size: 12px; margin-top: 5px;">
                                   ✓ Read on {book.get('date_read') or 'N/A'}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        if st.button(f"View Details", key=f"read_{book['book_id']}", use_container_width=True):
                            with st.spinner(f'Fetching details for "{book["title"]}"...'):
                                api_rating = book.get('average_rating') or book.get('rating')
                                book_details = get_book_details_from_google(
                                    book['title'],
                                    book['author'],
                                    book.get('isbn'),
                                    api_rating=api_rating
                                )
                                book_details['book_id'] = book['book_id']
                                book_details['already_read'] = True
                                st.session_state.selected_book = book_details
                                st.session_state.view_mode = 'book_details'
                            st.rerun()
    else:
        st.info("You haven't marked any books as read yet.")

# SEARCH VIEW 
elif st.session_state.view_mode == 'search':
    st.title("Search Books")
    st.markdown("### Find books to add to your reading list")

    if st.session_state.current_user:
        st.session_state.read_books = get_read_books(st.session_state.current_user)

    read_books_set = {
        (book.get('title', '').lower().strip(), book.get('author', '').lower().strip())
        for book in st.session_state.read_books
    }

    with st.form(key="search_form"):
        search_query = st.text_input(
            "Search by title:",
            placeholder="Enter book title ...",
            key="search_input"
        )
        search_button = st.form_submit_button("Search", type="primary", use_container_width=True)

    if search_button and search_query.strip():
        results = search_books(search_query.strip(), st.session_state.books_database)
        st.session_state.search_results = results
        st.session_state.last_search_query = search_query.strip()
    elif search_button and not search_query.strip():
        st.warning("Please enter a search query.")
        st.session_state.search_results = []

    if st.session_state.search_results:
        st.success(f"**{len(st.session_state.search_results)} books found** for '{st.session_state.get('last_search_query', '')}'")

        for idx, book in enumerate(st.session_state.search_results[:10]):
            book_title = book.get('title', '').lower().strip()
            book_author = book.get('author', '').lower().strip()
            is_read = (book_title, book_author) in read_books_set

            col1, col2 = st.columns([4, 1])
            with col1:
                if is_read:
                    st.markdown(f"**{book.get('title', 'Unknown')}** by *{book.get('author', 'Unknown')}* - ⭐ {format_rating(book.get('rating'))} *(Already Read)*")
                else:
                    st.markdown(f"**{book.get('title', 'Unknown')}** by *{book.get('author', 'Unknown')}* - ⭐ {format_rating(book.get('rating'))}")
            with col2:
                if st.button("View", key=f"search_view_{idx}_{book.get('book_id', idx)}", use_container_width=True):
                    with st.spinner(f'Fetching details...'):
                        api_rating = book.get('rating') or book.get('average_rating')
                        book_details = get_book_details_from_google(
                            book.get('title', ''),
                            book.get('author', ''),
                            book.get('isbn'),
                            api_rating=api_rating
                        )
                        book_details['book_id'] = book.get('book_id', '')
                        book_details['already_read'] = is_read
                        st.session_state.selected_book = book_details
                        st.session_state.previous_view = 'search'
                        st.session_state.view_mode = 'book_details'
                    st.rerun()
            st.divider()
    else:
        st.info("Enter a search query and click 'Search' to find books.")

# BOOK DETAILS VIEW (Keep your existing code - it's already good!)
elif st.session_state.view_mode == 'book_details':
    if st.button("← Back"):
        st.session_state.view_mode = st.session_state.get('previous_view', 'recommendations')
        st.session_state.show_rating_modal = False
        st.rerun()

    book = st.session_state.selected_book
    already_read = book.get('already_read', False)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(book['cover_url'], width=250)
        st.markdown("**Book Information**")
        st.markdown(f"📅 **Published:** {book['published_date']}")
        st.markdown(f"📄 **Pages:** {book['page_count']}")
        st.markdown(f"🌐 **Language:** {book['language'].upper()}")
        if book['preview_link'] != '#':
            st.markdown(f"[📖 Preview on Google Books]({book['preview_link']})")

    with col2:
        st.title(book['title'])
        st.markdown(f"**by {book['author']}**")
        if book['rating'] != 'N/A':
            st.markdown(f"⭐ **{book['rating']}/5** ({book['ratings_count']} ratings)")
        else:
            st.markdown("⭐ Rating not available")

        if book['categories']:
            st.markdown("**Categories:**")
            genre_html = "".join([f'<span class="genre-tag">{genre}</span>' for genre in book['categories']])
            st.markdown(genre_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Description")
    st.markdown(f'<div class="description-text">{book["description"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    if already_read:
        st.success("✓ You've already read this book!")
    else:
        if st.session_state.show_rating_modal and st.session_state.rating_book_info:
            st.markdown("---")
            st.markdown("#### ⭐ Rate this book before marking as read")

            rating = st.slider(
                "How would you rate this book?",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                format="%d ⭐",
                key="rating_slider"
            )

            stars_display = "⭐" * rating + "☆" * (5 - rating)
            st.markdown(f"**Your rating:** {stars_display} ({rating}/5)")

            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.button("Submit Rating & Mark as Read", type="primary", use_container_width=True):
                    try:
                        payload = {
                            "user_id": st.session_state.current_user,
                            "book_id": st.session_state.rating_book_info['book_id'],
                            "book_title": st.session_state.rating_book_info.get('title', ''),
                            "rating": rating
                        }

                        response = requests.post(
                            f"{API_BASE_URL}/mark-read",
                            json=payload,
                            timeout=10
                        )

                        if response.status_code == 200:
                            st.success(f"'{st.session_state.rating_book_info['title']}' marked as read with {rating}⭐ rating!")
                            st.balloons()
                            st.session_state.selected_book['already_read'] = True
                            st.session_state.read_books = get_read_books(st.session_state.current_user)
                            st.session_state.show_rating_modal = False
                            st.session_state.rating_book_info = None
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"API Error: {response.status_code} - {response.text}")

                    except requests.exceptions.Timeout:
                        st.error("Request timed out. Please try again.")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the server. Please check your connection.")
                    except Exception as e:
                        st.error(f"Error marking book as read: {str(e)}")

            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_rating_modal = False
                    st.session_state.rating_book_info = None
                    st.rerun()
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("❤️", key="like_btn",use_container_width=True,help='Add to Favourites'):
                    send_click_event(st.session_state.current_user, book['book_id'], "like", book['title'])
                    st.success("Favorite Recorded")

            with col2:
                if st.button("Add to List", key="reading_list_btn",use_container_width=True):
                    send_click_event(st.session_state.current_user, book['book_id'], "add_to_list", book['title'])
                    st.success("Added to your reading list!")

            with col3:
                if st.button("✓ Mark as Read", key="mark_read_btn",use_container_width=True):
                    st.session_state.show_rating_modal = True
                    st.session_state.rating_book_info = {
                        'book_id': book['book_id'],
                        'title': book['title']
                    }
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Book Recommendation System | MLOps Project</p>
        <p><small>Mock recommendations - Real book data from Google Books API</small></p>
    </div>
""", unsafe_allow_html=True)