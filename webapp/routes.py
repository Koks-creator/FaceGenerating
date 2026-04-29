import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import base64
import random
import string
from io import BytesIO
from PIL import Image
from functools import wraps
from time import time
from flask import render_template, flash, request
import numpy as np
import requests
import cv2

from webapp import app, forms
from config import Config

_api_status_cache = {"connected": False, "last_check": 0}
API_CHECK_INTERVAL = Config.WEB_API_CHECK_INTERVAL

# Exceptions
class FailedRequest(Exception):
    pass

####

def random_string(n: int = 10) -> str:
    pool = string.ascii_lowercase + "".join(str(i) for i in range(10))
    return "".join([random.choice(pool) for _ in range(n)])


def check_api_connection() -> bool:
    """checks connection to api with some cache"""
    now = time()
    
    if now - _api_status_cache["last_check"] < API_CHECK_INTERVAL:
        return _api_status_cache["connected"]
    
    try:
        response = requests.get(
            f"http://{Config.API_HOST}:{Config.API_PORT}/health",
            timeout=5
        )
        connected = response.status_code == 200 and response.json().get("status") == "all green"
        
    except Exception as e:
        app.logger.error(f"API connection check failed: {e}")
        connected = False
    
    _api_status_cache["connected"] = connected
    _api_status_cache["last_check"] = now
    return connected

def require_api_connection(f) -> None:
    """Dekorator sprawdzający połączenie z API."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not check_api_connection():
            flash("API is not available. Try again later.", "danger")
            return render_template("home.html", form=forms.MainForm(), results=[])
        return f(*args, **kwargs)
    return decorated_function

def base64_to_numpy(base64_str):
    image_data = base64.b64decode(base64_str)
    buffer = BytesIO(image_data)
    img = Image.open(buffer)
    return np.array(img)


@app.route("/health", methods=["GET"])
def health_check():
    api_status = check_api_connection()
    return {"status": api_status}


@app.route("/", methods=["GET", "POST"])
@require_api_connection
def home():
    form = forms.MainForm()
    file_names = []

    form_validation = form.validate_on_submit()
    print(form_validation)
    if form_validation:
        try:
            gen_num = form.gen_num_field.data
            model_name = form.models_list_field.data

            ## GET TOKEN
            token_req = requests.post(
                f"http://{Config.API_HOST}:{Config.API_PORT}/token",
                data={
                    "username": Config.WEBAPP_USER_LOGIN,
                    "password": Config.WEBAPP_USER_PASSWORD
                }
            )
            if token_req.status_code != 200 or not token_req.json().get("access_token"):
                app.logger.error(f"Failed to get a token, {token_req.status_code}, {token_req.content}")
                raise FailedRequest(f"Failed to get a token, {token_req.status_code}, {token_req.content}")

            token = token_req.json()["access_token"]

            ## REQUEST FACE GENERATION
            headers = {"Authorization": f"Bearer {token}"}
            req = requests.post(f"http://{Config.API_HOST}:{Config.API_PORT}/generate_faces", 
                json={
                    "model_name": model_name,
                    "gen_num": gen_num
                },
                headers=headers
            )
            if req.status_code != 200:
                app.logger.error(f"Failed to generate faces, {req.status_code}, {req.content}")
                raise FailedRequest(f"Failed to generate faces, {req.status_code}, {req.content}")
            images = req.json()["images"]

            ## RETURN IMAGES
            for ind, img in enumerate(images):
                num_img = base64_to_numpy(img)
                num_img = cv2.cvtColor(num_img, cv2.COLOR_BGR2RGB)
                random_str = random_string(n=10)
                file_name = f"{int(time())}_{ind}_{random_str}_face.png" # wiecej losowosci
                file_names.append(file_name)
                file_path = Config.WEB_APP_TEMP_UPLOADS_FOLDER / file_name

                cv2.imwrite(file_path, num_img)
        except FailedRequest as e:
            flash(e)
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as e:
            flash(f"Connection to API failed: {e}")
        except Exception as e:
            flash(f"Unknown error: {e}")
    return render_template("home.html", form=form, file_names=file_names)