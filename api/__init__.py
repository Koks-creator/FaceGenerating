import sys
from pathlib import Path
import os
from logging import Logger, getLogger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI
import json
from database.database import Base, engine

from config import Config
from custom_logger import CustomLogger
from face_generator import FaceGenerator
from database.database import SessionLocal
from database.models import User, UserRole
from auth.auth import hash_password

try:
    def setup_logging() -> Logger:
        """Configure logging for the api"""
        log_dir = os.path.dirname(Config.API_LOG_FILE)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger = CustomLogger(
            logger_name="middleware_logger",
            logger_log_level=Config.CLI_LOG_LEVEL,
            file_handler_log_level=Config.FILE_LOG_LEVEL,
            log_file_name=Config.API_LOG_FILE
        ).create_logger()

        return logger

    logger = getLogger("middleware_logger")

    if not logger.handlers:
        logger = setup_logging()
    logger.info("Starting api")

    # Starting db
    db_dir = os.path.dirname(Config.DATABASE_URL.replace("sqlite:///", ""))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    Base.metadata.create_all(bind=engine)
    logger.info("DB ready")

    # Satup
    users_to_create = {
        "admin": {
            "username": Config.ADMIN_LOGIN,
            "password": Config.ADMIN_PASSWORD,
            "role": "admin"
        },
         "webapp": {
            "username": Config.WEBAPP_USER_LOGIN,
            "password": Config.WEBAPP_USER_PASSWORD,
            "role": "user"
        }
    }
    
    # osobny try?
    with SessionLocal() as db:
        for user_name, user_conf in users_to_create.items():
            existing = db.query(User).filter(User.username == user_name).first()
            if existing:
                logger.info(f"{user_name} account already created")
            else:
                logger.info(f"No {user_name} account, creating...")
                user = User(
                    username=user_conf["username"],
                    hashed_password=hash_password(user_conf["password"]),
                    role=UserRole(user_conf["role"])
                )
                db.add(user)
                db.commit()

    # Starting api
    app = FastAPI(title="FaceGeneratorApi")
    with open(Config.API_MODELS_LIST_PATH) as f:
        models_to_load = json.load(f)

    logger.info("Loading models")
    loaded_models = {}
    for model_name, model_conf in models_to_load.items():
        model_file = model_conf["filename"]
        load_attention = model_conf["load_attention"]
        try:
            loaded_models[model_name] = FaceGenerator(
                model_path=Path(Config.MODELS_PATH) / model_file,
                load_attention=load_attention
            )
            logger.info(f"{model_name} loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {model_name}, {e}", exc_info=True)

    logger.info(f"Loaded models: \n{loaded_models}")

    from api import routes
    logger.info("Api initialized")
except Exception as e:
    logger.error(f"Error on init: {e}", exc_info=True)
    raise e