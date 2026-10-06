"""
Auth routes: register, login, and a get_current_user dependency other
routers can import to protect their own endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app_services.auth_service import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from database import Base, engine, get_db
from models_auth import User
from models import Household
from models_profile import UserProfile

Base.metadata.create_all(bind=engine)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    household_id: int
    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    return user


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )

    claimed_ids = db.query(User.household_id)
    next_household = (
        db.query(Household.household_id)
        .filter(~Household.household_id.in_(claimed_ids))
        .order_by(Household.household_id)
        .first()
    )

    if next_household is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No unclaimed households are available right now.",
        )

    new_user = User(
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        household_id=next_household.household_id,
    )
    db.add(new_user)

    try:
        db.flush()  # gives new_user an id, and surfaces a clash before anything is saved
        # Registering = a new user. (An account with no profile row counts as an existing user.)
        db.add(UserProfile(user_id=new_user.id, user_type="new"))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Household assignment conflict, please try registering again.",
        )

    db.refresh(new_user)
    return new_user


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.email})
    return Token(access_token=access_token)


@router.get("/me", response_model=UserOut)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user