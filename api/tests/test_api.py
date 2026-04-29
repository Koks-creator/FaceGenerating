import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pytest
import json
from fastapi.testclient import TestClient

from api import app
from database.database import SessionLocal
from database.models import User, UserRole
from auth.auth import hash_password
from config import Config
from custom_logger import CustomLogger

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=Path(Config.ROOT_PATH) / "api" / "tests" / "test_logs.log"
).create_logger()

TEST_USER = {"username": Config.TEST_USER_LOGIN, "password": Config.TEST_USER_PASSWORD}
ADMIN_USER = {"username": Config.ADMIN_LOGIN, "password": Config.ADMIN_PASSWORD} # probably stupid ahh idea

def _ensure_user(username: str, password: str, role: UserRole):
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            user = User(
                username=username,
                hashed_password=hash_password(password),
                role=role,
            )
            db.add(user)
            db.commit()
            logger.info(f"Created test user: {username} ({role.value})")
        elif user.role != role:
            # na wypadek gdyby ktoś ręcznie zmienił rolę
            user.role = role
            db.commit()

# ----------------- FIXTURES -----------------
@pytest.fixture(scope="session", autouse=True)
def log_test_session():
    logger.info("Starting tests")
    _ensure_user(TEST_USER["username"], TEST_USER["password"], UserRole.USER)
    yield
    logger.info("Finishing tests")

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def user_headers(client):
    token = _get_token(client, TEST_USER)
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def admin_headers(client):
    token = _get_token(client, ADMIN_USER)
    return {"Authorization": f"Bearer {token}"}

def _get_token(client: TestClient, credentials: dict) -> str:
    resp = client.post(
        "/token",
        data={**credentials}
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]

# ----------------- BASIC ENDPOINTS -----------------
def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    logger.info("test_root_endpoint passed")

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.json()["status"] == "all green"
    logger.info("test_health_endpoint passed")

# ----------------- AUTH -----------------
def test_login_success(client):
    response = client.post("/token", data=TEST_USER)
    assert response.status_code == 200
    assert "access_token" in response.json()
    logger.info("test_login_success passed")

def test_login_failed(client):
    response = client.post("/token", data={
        "username": Config.TEST_USER_LOGIN, "password": "dupen"
    })
    assert response.status_code == 401
    assert "access_token" not in response.json()
    logger.info("test_login_failed passed")


# ----------------- PROTECTED ENDPOINTS -----------------
def test_list_models_endpoint(client):
    response = client.get("/list_models")
    with open(Path(Config.ROOT_PATH) / "api" / "model_to_load.json") as f:
        models_to_load = json.load(f)
    assert response.json() == list(models_to_load.keys())
    logger.info("test_list_models_endpoint passed")

def test_generate_faces(client, user_headers):
    models = ["dcgan", "wganv4"]

    for model in models:
        response = client.post("/generate_faces", json={
            "model_name": model,
            "gen_num": 2

            },
            headers=user_headers
        )
        data = response.json()
        assert response.status_code == 200
        assert len(data["images"]) == 2
        assert isinstance(data["images"], list)
        assert all(isinstance(img, str) for img in data["images"])
    logger.info("test_generate_faces passed")

def test_list_users_endpoint(client, admin_headers):
    response = client.get(
         "/list/users?filter=login:eq:admin|login:eq:webapp|login:eq:test_user",
         headers=admin_headers
    )
    data = response.json()
    assert response.status_code == 200
    assert [d["username"] for d in data] == ["admin", "webapp", "test_user"]