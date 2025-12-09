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
            
            // Function to send vitals to backend
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
            
            // Calculate INP from interaction entries (P98 of all interactions)
            function calculateINP() {{
                const entries = parentWin._inpEntries;
                if (entries.length === 0) return;
                
                // Sort by duration descending
                entries.sort((a, b) => b - a);
                
                // INP is the P98 (or worst if fewer than 50 interactions)
                const p98Index = Math.min(Math.floor(entries.length * 0.02), entries.length - 1);
                parentWin._webVitals.inp = Math.round(entries[p98Index]);
            }}
            
            // Use PerformanceObserver for more reliable metrics in Streamlit
            try {{
                // LCP Observer
                new PerformanceObserver((list) => {{
                    const entries = list.getEntries();
                    const lastEntry = entries[entries.length - 1];
                    parentWin._webVitals.lcp = Math.round(lastEntry.startTime);
                    sendVitals();
                }}).observe({{type: 'largest-contentful-paint', buffered: true}});
                
                // CLS Observer - accumulate in parent window to persist across Streamlit rerenders
                new PerformanceObserver((list) => {{
                    for (const entry of list.getEntries()) {{
                        if (!entry.hadRecentInput) {{
                            parentWin._clsValue += entry.value;
                        }}
                    }}
                    parentWin._webVitals.cls = parseFloat(parentWin._clsValue.toFixed(4));
                    sendVitals();
                }}).observe({{type: 'layout-shift', buffered: true}});
                
                // FCP Observer
                new PerformanceObserver((list) => {{
                    const entries = list.getEntries();
                    if (entries.length > 0) {{
                        parentWin._webVitals.fcp = Math.round(entries[0].startTime);
                        sendVitals();
                    }}
                }}).observe({{type: 'paint', buffered: true}});
                
                // INP Observer (Interaction to Next Paint) - lowered threshold to 0 to capture all interactions
                new PerformanceObserver((list) => {{
                    for (const entry of list.getEntries()) {{
                        if (entry.duration > 0) {{
                            parentWin._inpEntries.push(entry.duration);
                            calculateINP();
                        }}
                    }}
                    sendVitals();
                }}).observe({{type: 'event', buffered: true, durationThreshold: 0}});
                
                // TTFB from Navigation Timing
                const navEntry = performance.getEntriesByType('navigation')[0];
                if (navEntry) {{
                    parentWin._webVitals.ttfb = Math.round(navEntry.responseStart);
                }}
                
                console.log('[Web Vitals] Monitoring initialized for session:', sessionId);
                
            }} catch (e) {{
                console.warn('[Web Vitals] PerformanceObserver not fully supported:', e);
            }}
            
            // Explicit interaction tracking for common user actions
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
            
            // Attach interaction listeners to parent window for comprehensive tracking
            ['click', 'keydown', 'pointerdown'].forEach(eventType => {{
                parentWin.document.addEventListener(eventType, trackInteraction, {{passive: true, capture: true}});
            }});
            
            // Also track in iframes (Streamlit components)
            try {{
                document.addEventListener('click', trackInteraction, {{passive: true, capture: true}});
                document.addEventListener('keydown', trackInteraction, {{passive: true, capture: true}});
            }} catch (e) {{}}
            
            // Force CLS measurement on dynamic content by observing mutations
            const mutationObserver = new MutationObserver(() => {{
                // Trigger a layout shift check after DOM changes
                if (parentWin._webVitals.cls === undefined) {{
                    parentWin._webVitals.cls = 0;
                }}
            }});
            mutationObserver.observe(parentWin.document.body, {{
                childList: true,
                subtree: true,
                attributes: true
            }});
            
            // Send on various triggers
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

        # Strategy 1: Try ISBN first if available
        if isbn:
            url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    book_data = data['items'][0]['volumeInfo']

        # Strategy 2: If ISBN failed or not available, use title + author with field-specific search
        if not book_data:
            # Use intitle: and inauthor: for precise matching
            encoded_title = quote(title)
            encoded_author = quote(author)
            url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{encoded_title}+inauthor:{encoded_author}"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    # Find best match - prioritize exact title match
                    best_match = None
                    for item in data['items']:
                        item_info = item.get('volumeInfo', {})
                        item_title = item_info.get('title', '').lower()
                        item_authors = [a.lower() for a in item_info.get('authors', [])]

                        # Check for exact title match
                        if title.lower() == item_title:
                            # Check if author matches
                            if any(author.lower() in a or a in author.lower() for a in item_authors):
                                best_match = item_info
                                break

                        # Store first result as fallback
                        if best_match is None:
                            best_match = item_info

                    book_data = best_match

        # Strategy 3: Fallback to simple combined search
        if not book_data:
            query = f"{title} {author}".replace(" ", "+")
            url = f"https://www.googleapis.com/books/v1/volumes?q={quote(query, safe='+')}"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    book_data = data['items'][0]['volumeInfo']

        # Process the found book data
        if book_data:
            # Prioritize API rating, fall back to Google's rating
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
        "event_type": event_type,  # "view", "click", "like", "add_to_list"
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
    st.session_state.view_mode = 'user_selection'  # 'user_selection', 'recommendations', 'book_details', 'read_books', 'search'
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
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .book-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 10px 0;
        transition: transform 0.2s;
        background-color: #f9f9f9;
    }
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .book-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
        color: #2c3e50;
    }
    .book-author {
        color: #666;
        margin-bottom: 5px;
        font-style: italic;
    }
    .rating {
        color: #f39c12;
        font-size: 16px;
    }
    .detail-section {
        margin: 20px 0;
    }
    .genre-tag {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px;
        background-color: #3498db;
        color: white;
        border-radius: 15px;
        font-size: 14px;
    }
    .description-text {
        line-height: 1.6;
        text-align: justify;
    }
    .sidebar-button {
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Inject Web Vitals monitoring (runs once per page load)
inject_web_vitals_monitoring(st.session_state.session_id, API_BASE_URL)

# Sidebar with navigation
with st.sidebar:
    st.title("Navigation")

    if st.session_state.api_call_count > 0:
        avg_latency = st.session_state.total_api_latency / st.session_state.api_call_count

        latency_color = "green" if avg_latency < 500 else "orange" if avg_latency < 1000 else "red"
        st.markdown(f"""
            <div style='background: rgba(0,0,0,0.05); padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                <div style='font-size: 12px; color: #666;'>API Performance</div>
                <div style='font-size: 18px; font-weight: bold; color: {latency_color};'>{avg_latency:.0f} ms</div>
                <div style='font-size: 11px; color: #888;'>{st.session_state.api_call_count} calls</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.current_user:
        st.markdown(f"**User:** User-{st.session_state.current_user[:4]}")
        st.markdown("---")

        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            if current_user:
                st.session_state.recommendations = get_recommendations(current_user)
                for book in st.session_state.recommendations:
                    send_click_event(current_user, book["book_id"], "view", book["title"])

            st.session_state.view_mode = "recommendations"
            st.session_state.show_rating_modal = False
            st.rerun()

        if st.button("📖 My Read Books", use_container_width=True, key="nav_read"):
            st.session_state.read_books = get_read_books(st.session_state.current_user)
            st.session_state.view_mode = 'read_books'
            st.session_state.show_rating_modal = False
            st.rerun()

        if st.button("🔍 Search Books", use_container_width=True, key="nav_search"):
            st.session_state.view_mode = 'search'
            st.session_state.show_rating_modal = False
            st.rerun()

        st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True, key="nav_logout"):
            st.session_state.current_user = None
            st.session_state.view_mode = 'user_selection'
            st.session_state.show_rating_modal = False
            st.rerun()
    else:
        st.info("Please login to access all features")


# Helper function to display book card
def display_book_card(book: Dict, col, button_prefix: str = "btn"):
    with col:
        st.markdown(f"""
            <div class="book-card">
                <div class="book-title">{book['title']}</div>
                <div class="book-author">by {book['author']}</div>
                <div class="rating">⭐ {format_rating(book.get('predicted_rating'))}</div>
            </div>
        """, unsafe_allow_html=True)

        if st.button(f"View Details", key=f"{button_prefix}_{book['book_id']}", use_container_width=True):
            send_click_event(
                st.session_state.current_user,
                book['book_id'],
                "click",
                book['title']
            )

            with st.spinner(f'Fetching details for "{book["title"]}"...'):
                # Get API rating
                api_rating = book.get('predicted_rating') or book.get('average_rating') or book.get('rating')

                book_details = get_book_details_from_google(
                    book['title'],
                    book['author'],
                    book.get('isbn'),
                    api_rating=api_rating  # Pass API rating as priority
                )
                book_details['book_id'] = book['book_id']

                st.session_state.selected_book = book_details
                st.session_state.previous_view = 'recommendations'
                st.session_state.view_mode = 'book_details'

            st.rerun()


# Main application logic
if st.session_state.view_mode == 'user_selection':
    st.title("📚 Book Recommendation System")
    st.markdown("### Select or Enter User ID")

    # Admin button at the top right
    col_title, col_admin = st.columns([5, 1])
    with col_admin:
        if st.button("🔐 Admin", use_container_width=True):
            st.session_state.show_admin_modal = True
            st.rerun()

    # Admin login modal
    if st.session_state.show_admin_modal:
        with st.container():
            st.markdown("---")
            st.markdown("### 🔐 Admin Login")

            admin_username = st.text_input("Username:", key="admin_user")
            admin_password = st.text_input("Password:", type="password", key="admin_pass")

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("Login", type="primary", use_container_width=True):
                    if admin_username == "admin" and admin_password == "admin":
                        st.success("Login successful! Redirecting to monitoring dashboard...")
                        admin_url = f"{API_BASE_URL}/report"
                        st.markdown(f'<meta http-equiv="refresh" content="1;url={admin_url}" />',
                                    unsafe_allow_html=True)
                        st.markdown(f"[Click here if not redirected]({admin_url})")
                    else:
                        st.error("Invalid credentials!")

            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_admin_modal = False
                    st.rerun()

            st.markdown("---")

    # User selection/input
    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_input("Enter User ID:", placeholder="user_12345")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            if user_input:
                st.session_state.current_user = user_input
                st.session_state.recommendations = get_recommendations(user_input)
                st.session_state.view_mode = 'recommendations'

                # Track that user viewed recommendations
                for book in st.session_state.recommendations:
                    send_click_event(user_input, book['book_id'], "view", book['title'])

                st.rerun()
            else:
                st.error("Please enter a user ID")

    # Quick select users
    st.markdown("### Or select a sample user:")
    quick_users = {
        "User-4b1af": "4b1af908229844ec02bc4b40aa6ea4dd",
        "User-2faa2": "2faa2ef7e9062a7339ed1e4299c7ecaf",
        "User-4597b": "4597ba0bb52054eae1e87534c78b13b8"
    }
    cols = st.columns(3)
    for idx, (display_name, user_id) in enumerate(quick_users.items()):
        with cols[idx]:
            if st.button(display_name, use_container_width=True):
                st.session_state.current_user = user_id
                st.session_state.recommendations = get_recommendations(user_id)
                st.session_state.view_mode = 'recommendations'

                for book in st.session_state.recommendations:
                    send_click_event(user_id, book['book_id'], "view", book['title'])

                st.rerun()

elif st.session_state.view_mode == 'recommendations':
    st.title(f"Recommendations for User-{st.session_state.current_user[:4]}")
    st.markdown("### Top 10 Books Just for You")

    # Display books in a grid (2 columns)
    if st.session_state.recommendations:
        for i in range(0, len(st.session_state.recommendations), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(st.session_state.recommendations):
                    book = st.session_state.recommendations[i + j]
                    display_book_card(book, cols[j], "rec")
    else:
        st.info("No recommendations available for this user.")

elif st.session_state.view_mode == 'read_books':
    st.title(f"📖 My Read Books")
    st.markdown(f"### Books you've already read")

    # Display read books
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

elif st.session_state.view_mode == 'search':
    st.title("🔍 Search Books")
    st.markdown("### Find books to add to your reading list")

    # Load user's read books if not already loaded
    if st.session_state.current_user:
        st.session_state.read_books = get_read_books(st.session_state.current_user)

    # Create set of read books using title + author (lowercase for matching)
    read_books_set = {
        (book.get('title', '').lower().strip(), book.get('author', '').lower().strip())
        for book in st.session_state.read_books
    }

    # Use a form to properly handle search
    with st.form(key="search_form"):
        search_query = st.text_input(
            "Search by title:",
            placeholder="Enter book title ...",
            key="search_input"
        )
        search_button = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

    # Process search when form is submitted
    if search_button and search_query.strip():
        results = search_books(search_query.strip(), st.session_state.books_database)
        st.session_state.search_results = results
        st.session_state.last_search_query = search_query.strip()
    elif search_button and not search_query.strip():
        st.warning("Please enter a search query.")
        st.session_state.search_results = []

    # Display results
    if st.session_state.search_results:
        st.success(f"**{len(st.session_state.search_results)} books found** for '{st.session_state.get('last_search_query', '')}'")

        for idx, book in enumerate(st.session_state.search_results[:10]):
            # Check if this book has been read using title + author
            book_title = book.get('title', '').lower().strip()
            book_author = book.get('author', '').lower().strip()
            is_read = (book_title, book_author) in read_books_set

            col1, col2 = st.columns([4, 1])
            with col1:
                if is_read:
                    st.markdown(f"✅ **{book.get('title', 'Unknown')}** by *{book.get('author', 'Unknown')}* - ⭐ {format_rating(book.get('rating'))} *(Already Read)*")
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
        st.info("💡 Enter a search query and click 'Search' to find books.")

elif st.session_state.view_mode == 'book_details':
    # Back button
    if st.button("← Back"):
        st.session_state.view_mode = st.session_state.get('previous_view', 'recommendations')
        st.session_state.show_rating_modal = False
        st.rerun()

    book = st.session_state.selected_book
    already_read = book.get('already_read', False)

    # Book details view
    col1, col2 = st.columns([1, 3])

    with col1:
        # Display cover image from Google Books with fixed smaller size
        st.image(book['cover_url'], width=250)

        # Additional info box
        st.markdown("**Book Information**")
        st.markdown(f"📅 **Published:** {book['published_date']}")
        st.markdown(f"📄 **Pages:** {book['page_count']}")
        st.markdown(f"🌐 **Language:** {book['language'].upper()}")

        if book['preview_link'] != '#':
            st.markdown(f"[📖 Preview on Google Books]({book['preview_link']})")

    with col2:
        st.title(book['title'])
        st.markdown(f"**by {book['author']}**")

        # Rating information
        if book['rating'] != 'N/A':
            st.markdown(f"⭐ **{book['rating']}/5** ({book['ratings_count']} ratings)")
        else:
            st.markdown("⭐ Rating not available")

        # Categories/Genres
        if book['categories']:
            st.markdown("**Categories:**")
            genre_html = "".join([f'<span class="genre-tag">{genre}</span>' for genre in book['categories']])
            st.markdown(genre_html, unsafe_allow_html=True)

    # Description section
    st.markdown("---")
    st.markdown("### 📖 Description")
    st.markdown(f'<div class="description-text">{book["description"]}</div>', unsafe_allow_html=True)

    # Action buttons
    st.markdown("---")

    if already_read:
        st.success("✅ You've already read this book!")
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

            # Display star rating visually
            stars_display = "⭐" * rating + "☆" * (5 - rating)
            st.markdown(f"**Your rating:** {stars_display} ({rating}/5)")

            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.button("✅ Submit Rating & Mark as Read", type="primary", use_container_width=True):
                    try:
                        # Prepare the payload
                        payload = {
                            "user_id": st.session_state.current_user,
                            "book_id": st.session_state.rating_book_info['book_id'],
                            "book_title": st.session_state.rating_book_info.get('title', ''),
                            "rating": rating
                        }

                        # Debug: Show what's being sent (remove in production)
                        # st.write("Sending payload:", payload)

                        response = requests.post(
                            f"{API_BASE_URL}/mark-read",
                            json=payload,
                            timeout=10
                        )

                        # Check response status
                        if response.status_code == 200:
                            st.success(
                                f"'{st.session_state.rating_book_info['title']}' marked as read with {rating}⭐ rating!")
                            st.balloons()

                            # Update the book's already_read status
                            st.session_state.selected_book['already_read'] = True

                            # Refresh read books list
                            st.session_state.read_books = get_read_books(st.session_state.current_user)

                            # Clear modal state
                            st.session_state.show_rating_modal = False
                            st.session_state.rating_book_info = None

                            # Rerun to update UI
                            time.sleep(1)  # Brief pause to show success message
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
            # Normal action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("❤️ Like this book", use_container_width=True):
                    send_click_event(
                        st.session_state.current_user,
                        book['book_id'],
                        "like",
                        book['title']
                    )
                    st.success("Preference recorded!")

            with col2:
                if st.button("📚 Add to Reading List", use_container_width=True):
                    send_click_event(
                        st.session_state.current_user,
                        book['book_id'],
                        "add_to_list",
                        book['title']
                    )
                    st.success("Added to your reading list!")

            with col3:
                if st.button("✅ Mark as Read", use_container_width=True):
                    # Show rating modal instead of directly marking as read
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