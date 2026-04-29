import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import List
from pydantic import BaseModel, Field, field_validator
import asyncio
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from sqlalchemy import or_, and_

from api import app, logger, loaded_models
from config import Config
from auth.auth import (
    Token,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_admin
)
from database.database import get_db
from database.models import User


OPERATORS = {
    "eq": lambda c, v: c == v,
    "gt": lambda c, v: c > v,
    "lt": lambda c, v: c < v,
    "gte": lambda c, v: c >= v,
    "lte": lambda c, v: c <= v,
    "like": lambda c, v: c.ilike(f"%{v}%"),
    "in": lambda c, v: c.in_(v.split(";")),
}

class GenerateRequest(BaseModel):
    model_name: str
    gen_num: int = Field(None, ge=Config.API_MODEL_MIN_GEN_LEN, le=Config.API_MODEL_MAX_GEN_LEN)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if v not in loaded_models:
            raise ValueError(f"Model '{v}' is not available, check available models using /list_models endpoint")
        return v
    
class GenerateResponse(BaseModel):
    images: List[str] # base64

class HealthResponse(BaseModel):
    status: str

class FilterUser(BaseModel):
    user_id: int


def numpy_to_base64(img_np: np.ndarray) -> str:
    img_np = (img_np * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(img_np)
    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.post("/token", response_model=Token, tags=["Auth"])
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.username})
    return Token(access_token=token, token_type="bearer")

@app.get("/")
async def alive():
    return "Hello, I'm alive :) https://www.youtube.com/watch?v=9DeG5WQClUI"

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    return HealthResponse(status="all green")

@app.get("/list_models")
async def list_models():
    return list(loaded_models.keys())

@app.post("/generate_faces", response_model=GenerateResponse)
async def generate_faces(body: GenerateRequest, current_user: User = Depends(get_current_user)):
    try:
        model_name = body.model_name
        gen_num = body.gen_num
        face_generator = loaded_models.get(model_name)
        if not face_generator:
            raise HTTPException(
                status_code=500,
                detail=f"No such model {model_name}, check available models using /list_models endpoint"
            )

        generated_images = await asyncio.to_thread(
            face_generator.generate_faces,
            gen_num
        )

        encoded = [numpy_to_base64(img) for img in generated_images]
        return GenerateResponse(
            images=encoded
        )
    except Exception as e:
        logger.error(f"Failed to generate faces, unhandled error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate faces, unhandled error: {e}"
        )


def apply_filters(query, model, filter_string: str):
    if not filter_string:
        return query

    and_parts = filter_string.split(",")

    conditions = []

    for part in and_parts:
        if "|" in part:
            # OR block
            or_conditions = []
            for sub in part.split("|"):
                field, op, value = sub.split(":")
                column = getattr(model, field, None)
                if column and op in OPERATORS:
                    or_conditions.append(OPERATORS[op](column, value))

            if or_conditions:
                conditions.append(or_(*or_conditions))
        else:
            field, op, value = part.split(":")
            column = getattr(model, field, None)

            if column and op in OPERATORS:
                conditions.append(OPERATORS[op](column, value))

    return query.filter(and_(*conditions))

def apply_sort(query, model, sort_string: str):
    if not sort_string:
        return query

    for field in sort_string.split(","):
        desc = field.startswith("-")
        field_name = field[1:] if desc else field

        column = getattr(model, field_name, None)
        if column:
            query = query.order_by(column.desc() if desc else column.asc())

    return query

# ?filter=login:eq:admin|login:eq:test_user

@app.get("/list/users/")
def list_users(
    request: Request,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    query = db.query(User)

    filter_str = request.query_params.get("filter")
    sort_str = request.query_params.get("sort")

    query = apply_filters(query, User, filter_str)
    query = apply_sort(query, User, sort_str)

    return query.all()