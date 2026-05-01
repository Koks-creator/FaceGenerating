from pathlib import Path
import json
from typing import Union
import os
import logging
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Overall
    ROOT_PATH: str = Path(__file__).resolve().parent

    # FOLDERS
    MODELS_PATH: Union[str, os.PathLike, Path] = Path(ROOT_PATH) / "models"

    # DATABASE
    DATABASE_URL: str = f"sqlite:///{Path(ROOT_PATH)}/data/app.db"

    # TOKENS, HASHING
    SECRET_KEY: str = os.getenv("SECRET_KEY")  # openssl rand -hex 32
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # SETUP USERS
    ADMIN_LOGIN: str = os.getenv("ADMIN_LOGIN")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD")
    WEBAPP_USER_LOGIN: str = os.getenv("WEB_USER")
    WEBAPP_USER_PASSWORD: str = os.getenv("WEB_USER_PASSWORD")

    # TEST USERS
    TEST_USER_LOGIN: str = os.getenv("TEST_USER_LOGIN")
    TEST_USER_PASSWORD: str = os.getenv("TEST_USER_PASSWORD")

    # API
    API_PORT: int = 5000
    API_HOST: str = "127.0.0.1"
    MAX_IMAGE_FILES: int = 5
    API_LOG_FILE: str = Path(ROOT_PATH) / "logs" / "api_logs.log"
    API_MODELS_LIST_PATH: Union[str, os.PathLike, Path] = Path(ROOT_PATH) / "api" / "model_to_load.json"
    API_MODEL_MIN_GEN_LEN: int = 1
    API_MODEL_MAX_GEN_LEN: int = 32

    # WEB APP
    WEB_APP_PORT: int = 8000
    WEB_APP_HOST: str = "127.0.0.1"
    WEB_APP_DEBUG: bool = True
    WEB_APP_LOG_FILE: str = Path(ROOT_PATH) / "logs" / "web_app.logs"
    WEB_APP_TEMP_UPLOADS_FOLDER =  Path(ROOT_PATH) / "webapp" / "static" / "temp_uploads"
    WEB_APP_FILES_LIFE_TIME: int = 60
    WEB_APP_USE_SSL: bool = False
    WEB_APP_SSL_FOLDER: str = f"{ROOT_PATH}/ocr_webapp/ssl_cert"
    WEB_APP_TESTING: bool = False
    WEB_APP_LOG_LEVEL: int = logging.DEBUG
    WEB_API_CHECK_INTERVAL: int = 10

    # LOGGER
    UVICORN_LOG_CONFIG_PATH: Union[str, os.PathLike, Path] = Path(ROOT_PATH) / "api" / "uvicorn_log_config.json"
    CLI_LOG_LEVEL: int = logging.DEBUG
    FILE_LOG_LEVEL: int = logging.DEBUG

    def get_uvicorn_logger(self) -> dict:
        with open(self.UVICORN_LOG_CONFIG_PATH) as f:
            log_config = json.load(f)
            log_config["handlers"]["file_handler"]["filename"] =  Path(self.ROOT_PATH) / "logs" / "api_logs.log"
            return log_config
