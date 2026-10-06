"""
Auth-related database models.
"""

from sqlalchemy import Column, Integer, BigInteger, String, DateTime, func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    # NOTE: no ForeignKey(...) here on purpose. households is registered
    # under models.py's own separate Base/metadata, not this one, so
    # create_all() can't resolve a cross-Base foreign key — it crashes
    # with NoReferencedTableError. Real-world validity is still enforced
    # in api/auth.py's register() function, which only ever assigns an
    # household_id that actually exists in the households table — so we
    # don't lose correctness, just the Postgres-level guarantee.
    household_id = Column(BigInteger, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())