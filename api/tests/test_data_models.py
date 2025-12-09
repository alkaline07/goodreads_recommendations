from api.data_models import (
    BookRecommendation,
    ClickRequest,
    LoginRequest,
    LoginResponse,
    MarkReadRequest,
)


def test_mark_read_defaults_to_zero_rating():
    request = MarkReadRequest(user_id="user-1", book_id=42)
    assert request.rating == 0


def test_login_response_contains_recommendations():
    rec = BookRecommendation(book_id=1, predicted_rating=4.5, title="Test", author="Author")
    response = LoginResponse(user_id="user-1", recommendations=[rec])
    assert response.user_id == "user-1"
    assert response.recommendations[0].title == "Test"


def test_click_request_optional_title():
    payload = ClickRequest(user_id="u", book_id=2, event_type="click")
    assert payload.book_title is None
    assert payload.event_type == "click"


def test_login_request_parses_user_id():
    payload = LoginRequest(user_id="abc")
    assert payload.user_id == "abc"

