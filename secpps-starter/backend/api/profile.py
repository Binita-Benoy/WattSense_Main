
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import get_current_user
from database import Base, engine, get_db
from models_auth import User
from models_profile import UserProfile

# models_profile must be imported BEFORE this line, or its tables aren't created.
Base.metadata.create_all(bind=engine)

router = APIRouter()


class ProfileCreate(BaseModel):
    user_type: Literal["new", "existing"]


def _as_dict(profile):
    if profile is None:
        return {"has_profile": False}
    return {"has_profile": True, "user_type": profile.user_type}


def _get(db: Session, user: User):
    return db.query(UserProfile).filter(UserProfile.user_id == user.id).first()


@router.post("/profile", status_code=status.HTTP_201_CREATED)
def create_profile(
    body: ProfileCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if _get(db, current_user):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You've already chosen. Use 'Change user type' to start over.",
        )
    profile = UserProfile(user_id=current_user.id, user_type=body.user_type)
    db.add(profile)
    db.commit()
    return _as_dict(profile)


@router.get("/profile/me")
def get_my_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _as_dict(_get(db, current_user))


@router.delete("/profile")
def reset_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = _get(db, current_user)
    if profile:
        db.delete(profile)
        db.commit()
    return {"has_profile": False}