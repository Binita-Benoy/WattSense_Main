"""
Auth-related database models.
"""

from sqlalchemy import Column, Integer, BigInteger, String, DateTime, ForeignKey, func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    household_id = Column(
        BigInteger, ForeignKey("households.household_id"), unique=True, nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())