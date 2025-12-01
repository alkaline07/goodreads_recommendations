import streamlit as st
import requests
from typing import List, Dict
import base64
from io import BytesIO


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
        "cover_url": "https://via.placeholder.com/400x600?text=No+Cover+Available",
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
        "event_type": event_type,  # "view" or "click"
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
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'user_selection'  # 'user_selection', 'recommendations', 'book_details'

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
    </style>
""", unsafe_allow_html=True)


# Helper function to display book card
def display_book_card(book: Dict, col):
    with col:
        st.markdown(f"""
            <div class="book-card">
                <div class="book-title">{book['title']}</div>
                <div class="book-author">by {book['author']}</div>
                <div class="rating">⭐ {book['rating']:.1f}</div>
            </div>
        """, unsafe_allow_html=True)

        if st.button(f"View Details", key=f"btn_{book['book_id']}", use_container_width=True):
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
    # Header with back button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"📚 Recommendations for {st.session_state.current_user}")
    with col2:
        if st.button("← Change User", use_container_width=True):
            st.session_state.view_mode = 'user_selection'
            st.rerun()

    st.markdown("### Top 10 Books Just for You")

    # Display books in a grid (2 columns)
    if st.session_state.recommendations:
        for i in range(0, len(st.session_state.recommendations), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(st.session_state.recommendations):
                    book = st.session_state.recommendations[i + j]
                    display_book_card(book, cols[j])
    else:
        st.info("No recommendations available for this user.")

elif st.session_state.view_mode == 'book_details':
    # Back button
    if st.button("← Back to Recommendations"):
        st.session_state.view_mode = 'recommendations'
        st.rerun()

    book = st.session_state.selected_book

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
        if st.button("🔗 Get Similar Books", use_container_width=True):
            mock_send_click_event(
                st.session_state.current_user,
                book['book_id'],
                "similar",
                book['title']
            )
            st.info("Feature coming soon!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Book Recommendation System | MLOps Project</p>
        <p><small>Mock recommendations - Real book data from Google Books API</small></p>
    </div>
""", unsafe_allow_html=True)