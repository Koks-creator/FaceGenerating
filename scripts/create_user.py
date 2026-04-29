import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import Union, List
from sqlalchemy.sql.elements import BinaryExpression

from database.database import SessionLocal, Base, engine
from database.models import User, UserRole
from auth.auth import hash_password

Base.metadata.create_all(bind=engine)


def list_data(db_filter: Union[None, BinaryExpression] = None) -> List[User]:
    with SessionLocal() as db:
        if db_filter:
            rows = db.query(User).filter(db_filter).all()
        else:
            rows = db.query(User).all()
    return rows

def create_user(username: str, password: str, role: str = "user", is_active: bool = True) -> bool:
    with SessionLocal() as db:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            print(f"User {username} already exists")
            return False
        
        user = User(
            username=username,
            hashed_password=hash_password(password),
            role=UserRole(role),
            is_active=is_active
        )
        db.add(user)
        db.commit()

        return True
    
def update_user(user_id: int, new_data: dict) -> bool:
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            return False
        
        for key, value in new_data.items():
            setattr(user, key, value)
        
        db.commit()
        return True
        

if __name__ == "__main__":
    # d = User.username == "admin"
    # x = create_user(username="Test", password="test", role="user")
    # x = list_data()
    # update_user(2, {
    #     "username": "test",
    # })
    x = list_data()
    print(x)
    # print(type(d))
# # parser.add_argument("username")
# # parser.add_argument("password")
# # args = parser.parse_args()

# db = SessionLocal()
# username = "dupen"
# password = "dupen"
# role = "user"
# try:
#     # existing = db.query(User).all()
#     # print(existing)
#     existing = db.query(User).filter(User.username == username).first()
#     if existing:
#         print(f"User '{username}' already exists")
#         sys.exit(1)

#     user = User(
#         username=username,
#         hashed_password=hash_password(password),
#         role=UserRole(role)
#     )
#     db.add(user)
#     db.commit()
#     print(f"Created user: {username}")
# finally:
#     db.close()