import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pytest
from unittest.mock import patch

from webapp import app

@pytest.fixture()
def app_client():
    app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
    })
    yield app

def test_home_page(app_client):
    response = app_client.test_client().get("/")

    assert response.status_code == 200

def test_health_when_api_up(app_client, monkeypatch):
    # faking api
    monkeypatch.setattr("webapp.routes.check_api_connection", lambda: True)
    resp = app_client.test_client().get("/health")
    assert resp.json == {"status": True}

def test_health_when_api_down(app_client, monkeypatch):
    monkeypatch.setattr("webapp.routes.check_api_connection", lambda: False)
    resp = app_client.test_client().get("/health")
    assert resp.json == {"status": False}


# mock_check ← @patch(...check_api_connection...)
# mock_post ← @patch(...requests.post...)
@patch("webapp.routes.requests.post")
@patch("webapp.routes.check_api_connection", return_value=True)
def test_gen_num_out_of_range(mock_check, mock_post, app_client):
    client = app_client.test_client()
    resp = client.post("/", data={"gen_num_field": 100, "models_list_field": "dcgan"})

    assert resp.status_code == 200
    mock_post.assert_not_called()
    assert b"face.png" not in resp.data

@patch("webapp.routes.requests.post")
@patch("webapp.routes.check_api_connection", return_value=True)
def test_invalid_model(mock_check, mock_post, app_client):
    client = app_client.test_client()
    resp = client.post("/", data={"gen_num_field": 2, "models_list_field": "dupen"})

    assert resp.status_code == 200
    mock_post.assert_not_called()
    assert b"face.png" not in resp.data