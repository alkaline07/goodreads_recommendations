import streamlit as st
import requests
from typing import List, Dict
import json
import os


# Mock data - Replace these functions with actual REST API calls
def mock_get_recommendations(user_id: str) -> List[Dict]:
    """Mock function to get top 10 book recommendations for a user"""
    # Replace this with: requests.get(f"{API_BASE_URL}/recommendations/{user_id}")
    mock_books = [
        {
            "book_id": "1",
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "rating": 4.5,
            "isbn": "9780743273565"
        },
        {
            "book_id": "2",
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee",
            "rating": 4.8,
            "isbn": "9780061120084"
        },
        {
            "book_id": "3",
            "title": "1984",
            "author": "George Orwell",
            "rating": 4.7,
            "isbn": "9780451524935"
        },
        {
            "book_id": "4",
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "rating": 4.6,
            "isbn": "9780141439518"
        },
        {
            "book_id": "5",
            "title": "The Catcher in the Rye",
            "author": "J.D. Salinger",
            "rating": 4.3,
            "isbn": "9780316769174"
        },
        {
            "book_id": "6",
            "title": "The Hobbit",
            "author": "J.R.R. Tolkien",
            "rating": 4.7,
            "isbn": "9780547928227"
        },
        {
            "book_id": "7",
            "title": "Harry Potter and the Sorcerer's Stone",
            "author": "J.K. Rowling",
            "rating": 4.8,
            "isbn": "9780439708180"
        },
        {
            "book_id": "8",
            "title": "The Lord of the Rings",
            "author": "J.R.R. Tolkien",
            "rating": 4.9,
            "isbn": "9780544003415"
        },
        {
            "book_id": "9",
            "title": "Animal Farm",
            "author": "George Orwell",
            "rating": 4.4,
            "isbn": "9780451526342"
        },
        {
            "book_id": "10",
            "title": "Brave New World",
            "author": "Aldous Huxley",
            "rating": 4.5,
            "isbn": "9780060850524"
        }
    ]
    return mock_books


def mock_get_read_books(user_id: str) -> List[Dict]:
    """Mock function to get user's read books"""
    # Replace this with: requests.get(f"{API_BASE_URL}/users/{user_id}/read_books")
    mock_read_books = [
        {
            "book_id": "read_1",
            "title": "The Alchemist",
            "author": "Paulo Coelho",
            "rating": 4.6,
            "isbn": "9780062315007",
            "date_read": "2024-01-15"
        },
        {
            "book_id": "read_2",
            "title": "Sapiens",
            "author": "Yuval Noah Harari",
            "rating": 4.7,
            "isbn": "9780062316097",
            "date_read": "2024-02-20"
        },
        {
            "book_id": "read_3",
            "title": "Educated",
            "author": "Tara Westover",
            "rating": 4.8,
            "isbn": "9780399590504",
            "date_read": "2024-03-10"
        }
    ]
    return mock_read_books


def load_books_database():
    """Load books database from JSON file"""
    import json
    import os

    try:
        # Try to load from JSON file
        json_path = 'data/books_database.json'
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

                # Transform the data to our required format
                books = []
                for item in raw_data:
                    books.append({
                        "book_id": item.get("book_id", ""),
                        "title": item.get("title_clean", "Unknown Title"),
                        "author": "Various Authors",  # Will be parsed from authors_flat if needed
                        "rating": float(item.get("average_rating", "3.5")),
                        "isbn": item.get("isbn_clean", "")
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
        {"book_id": "db_1", "title": "The Silent Patient", "author": "Alex Michaelides", "rating": 4.5,
         "isbn": "9781250301697"},
        {"book_id": "db_2", "title": "Where the Crawdads Sing", "author": "Delia Owens", "rating": 4.6,
         "isbn": "9780735219090"},
        {"book_id": "db_3", "title": "Atomic Habits", "author": "James Clear", "rating": 4.8, "isbn": "9780735211292"},
        {"book_id": "db_4", "title": "The Midnight Library", "author": "Matt Haig", "rating": 4.3,
         "isbn": "9780525559474"},
        {"book_id": "db_5", "title": "Project Hail Mary", "author": "Andy Weir", "rating": 4.7,
         "isbn": "9780593135204"},
        {"book_id": "db_6", "title": "Dune", "author": "Frank Herbert", "rating": 4.6, "isbn": "9780441172719"},
        {"book_id": "db_7", "title": "The Seven Husbands of Evelyn Hugo", "author": "Taylor Jenkins Reid",
         "rating": 4.7, "isbn": "9781501161933"},
        {"book_id": "db_8", "title": "Normal People", "author": "Sally Rooney", "rating": 4.2, "isbn": "9781984822178"},
        {"book_id": "db_9", "title": "The Nightingale", "author": "Kristin Hannah", "rating": 4.8,
         "isbn": "9780312577223"},
        {"book_id": "db_10", "title": "Circe", "author": "Madeline Miller", "rating": 4.6, "isbn": "9780316556347"},
        {"book_id": "db_11", "title": "The Book Thief", "author": "Markus Zusak", "rating": 4.7,
         "isbn": "9780375842207"},
        {"book_id": "db_12", "title": "The Kite Runner", "author": "Khaled Hosseini", "rating": 4.6,
         "isbn": "9781594631931"},
        {"book_id": "db_13", "title": "Life of Pi", "author": "Yann Martel", "rating": 4.3, "isbn": "9780156027328"},
        {"book_id": "db_14", "title": "The Hunger Games", "author": "Suzanne Collins", "rating": 4.5,
         "isbn": "9780439023481"},
        {"book_id": "db_15", "title": "Gone Girl", "author": "Gillian Flynn", "rating": 4.4, "isbn": "9780307588371"},
    ]


def search_books(query: str, books_db: List[Dict]) -> List[Dict]:
    """Search books by title or author"""
    if not query:
        return books_db[:10]  # Return first 10 if no query

    query_lower = query.lower()
    results = [
        book for book in books_db
        if query_lower in book['title'].lower() or query_lower in book['author'].lower()
    ]
    return results[:20]  # Return top 20 matches


def get_book_details_from_google(title: str, author: str, isbn: str = None) -> Dict:
    """Fetch book details from Google Books API"""
    try:
        # Try searching by ISBN first if available
        if isbn:
            url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        else:
            # Search by title and author
            query = f"{title} {author}".replace(" ", "+")
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}"

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if 'items' in data and len(data['items']) > 0:
                book_info = data['items'][0]['volumeInfo']

                # Extract relevant information
                return {
                    "title": book_info.get('title', title),
                    "author": ', '.join(book_info.get('authors', [author])),
                    "description": book_info.get('description', 'No description available.'),
                    "cover_url": book_info.get('imageLinks', {}).get('thumbnail',
                                                                     book_info.get('imageLinks', {}).get(
                                                                         'smallThumbnail',
                                                                         'https://via.placeholder.com/300x450?text=No+Cover')),
                    "published_date": book_info.get('publishedDate', 'Unknown'),
                    "page_count": book_info.get('pageCount', 'N/A'),
                    "categories": book_info.get('categories', ['General']),
                    "rating": book_info.get('averageRating', 'N/A'),
                    "ratings_count": book_info.get('ratingsCount', 'N/A'),
                    "language": book_info.get('language', 'en'),
                    "preview_link": book_info.get('previewLink', '#')
                }

        # Fallback if API call fails
        return create_fallback_book_details(title, author)

    except Exception as e:
        st.error(f"Error fetching book details: {str(e)}")
        return create_fallback_book_details(title, author)


def create_fallback_book_details(title: str, author: str) -> Dict:
    """Create fallback book details if Google Books API fails"""
    return {
        "title": title,
        "author": author,
        "description": "Description not available. Please try again later.",
        "cover_url": "https://via.placeholder.com/300x450?text=No+Cover+Available",
        "published_date": "Unknown",
        "page_count": "N/A",
        "categories": ["General"],
        "rating": "N/A",
        "ratings_count": "N/A",
        "language": "en",
        "preview_link": "#"
    }


def mock_send_click_event(user_id: str, book_id: str, event_type: str, book_title: str = None):
    """Mock function to send click/interaction events for CTR tracking"""
    # Replace this with: requests.post(f"{API_BASE_URL}/events", json={...})
    event_data = {
        "user_id": user_id,
        "book_id": book_id,
        "book_title": book_title,
        "event_type": event_type,  # "view", "click", "like", "add_to_list", "similar", "mark_read"
        "timestamp": "2024-01-01T00:00:00Z"
    }
    # In production: response = requests.post(f"{API_BASE_URL}/events", json=event_data)
    print(f"Event tracked: {event_data}")
    return {"status": "success"}


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

# Sidebar with navigation
with st.sidebar:
    st.title("📚 Navigation")

    if st.session_state.current_user:
        st.markdown(f"**User:** {st.session_state.current_user}")
        st.markdown("---")

        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            st.session_state.view_mode = 'recommendations'
            st.rerun()

        if st.button("📖 My Read Books", use_container_width=True, key="nav_read"):
            st.session_state.read_books = mock_get_read_books(st.session_state.current_user)
            st.session_state.view_mode = 'read_books'
            st.rerun()

        if st.button("🔍 Search Books", use_container_width=True, key="nav_search"):
            st.session_state.view_mode = 'search'
            st.rerun()

        st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True, key="nav_logout"):
            st.session_state.current_user = None
            st.session_state.view_mode = 'user_selection'
            st.rerun()
    else:
        st.info("Please login to access features")


# Helper function to display book card
def display_book_card(book: Dict, col, button_prefix: str = "btn"):
    with col:
        st.markdown(f"""
            <div class="book-card">
                <div class="book-title">{book['title']}</div>
                <div class="book-author">by {book['author']}</div>
                <div class="rating">⭐ {book['rating']:.1f}</div>
            </div>
        """, unsafe_allow_html=True)

        if st.button(f"View Details", key=f"{button_prefix}_{book['book_id']}", use_container_width=True):
            # Track click event
            mock_send_click_event(
                st.session_state.current_user,
                book['book_id'],
                "click",
                book['title']
            )

            # Show loading spinner while fetching from Google Books API
            with st.spinner(f'Fetching details for "{book["title"]}"...'):
                # Get book details from Google Books API
                book_details = get_book_details_from_google(
                    book['title'],
                    book['author'],
                    book.get('isbn')
                )
                book_details['book_id'] = book['book_id']

                st.session_state.selected_book = book_details
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
                        st.success("Login successful! Redirecting...")
                        # Redirect to admin panel URL (replace with your actual URL)
                        st.markdown('<meta http-equiv="refresh" content="1;url=https://admin-panel.example.com" />',
                                    unsafe_allow_html=True)
                        st.markdown("[Click here if not redirected](https://admin-panel.example.com)")
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
                st.session_state.recommendations = mock_get_recommendations(user_input)
                st.session_state.view_mode = 'recommendations'

                # Track that user viewed recommendations
                for book in st.session_state.recommendations:
                    mock_send_click_event(user_input, book['book_id'], "view", book['title'])

                st.rerun()
            else:
                st.error("Please enter a user ID")

    # Quick select users (mock)
    st.markdown("### Or select a sample user:")
    quick_users = ["user_001", "user_002", "user_003", "user_004"]
    cols = st.columns(4)
    for idx, user in enumerate(quick_users):
        with cols[idx]:
            if st.button(user, use_container_width=True):
                st.session_state.current_user = user
                st.session_state.recommendations = mock_get_recommendations(user)
                st.session_state.view_mode = 'recommendations'

                # Track views
                for book in st.session_state.recommendations:
                    mock_send_click_event(user, book['book_id'], "view", book['title'])

                st.rerun()

elif st.session_state.view_mode == 'recommendations':
    st.title(f"📚 Recommendations for {st.session_state.current_user}")
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
                                <div class="rating">⭐ {book['rating']:.1f}</div>
                                <div style="color: #27ae60; font-size: 12px; margin-top: 5px;">
                                    ✓ Read on {book.get('date_read', 'N/A')}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        if st.button(f"View Details", key=f"read_{book['book_id']}", use_container_width=True):
                            with st.spinner(f'Fetching details for "{book["title"]}"...'):
                                book_details = get_book_details_from_google(
                                    book['title'],
                                    book['author'],
                                    book.get('isbn')
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

    # Search bar with search button
    col1, col2 = st.columns([5, 1])
    with col1:
        search_query = st.text_input("Search by title or author:", placeholder="Enter book title or author name...",
                                     key="search_input")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Trigger search on button click or if query exists
    if search_query and (search_button or search_query):
        # Get matching books
        matching_books = search_books(search_query, st.session_state.books_database)

        if matching_books:
            st.markdown(f"**{len(matching_books)} books found**")

            # Show autocomplete-style results
            for book in matching_books[:10]:  # Show top 10 matches
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Display book info in a compact format
                    st.markdown(f"**{book['title']}** by *{book['author']}* - ⭐ {book['rating']}")
                with col2:
                    # Button to view details
                    if st.button("View", key=f"search_select_{book['book_id']}", use_container_width=True):
                        # Track click event
                        mock_send_click_event(
                            st.session_state.current_user,
                            book['book_id'],
                            "click",
                            book['title']
                        )

                        # Fetch and show book details
                        with st.spinner(f'Fetching details for "{book["title"]}"...'):
                            book_details = get_book_details_from_google(
                                book['title'],
                                book['author'],
                                book.get('isbn')
                            )
                            book_details['book_id'] = book['book_id']

                            st.session_state.selected_book = book_details
                            st.session_state.view_mode = 'book_details'

                        st.rerun()

                st.markdown("---")
        else:
            st.warning("No books found matching your search.")
    else:
        st.info("Enter a search query and click 'Search' to find books...")

elif st.session_state.view_mode == 'book_details':
    # Back button
    if st.button("← Back"):
        st.session_state.view_mode = 'recommendations'
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
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("👍 Like this book", use_container_width=True):
                mock_send_click_event(
                    st.session_state.current_user,
                    book['book_id'],
                    "like",
                    book['title']
                )
                st.success("Preference recorded!")

        with col2:
            if st.button("📚 Add to Reading List", use_container_width=True):
                mock_send_click_event(
                    st.session_state.current_user,
                    book['book_id'],
                    "add_to_list",
                    book['title']
                )
                st.success("Added to your reading list!")

        with col3:
            if st.button("✅ Mark as Read", use_container_width=True):
                mock_send_click_event(
                    st.session_state.current_user,
                    book['book_id'],
                    "mark_read",
                    book['title']
                )
                st.success("Marked as read!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Book Recommendation System | MLOps Project</p>
        <p><small>Mock recommendations - Real book data from Google Books API</small></p>
    </div>
""", unsafe_allow_html=True)