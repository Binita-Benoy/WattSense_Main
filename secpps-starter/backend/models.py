"""
ORM models matching infra/init-db/01-schema.sql (Alka's schema) exactly.
If she changes the schema, update this file to match — it's the single
shared definition everyone else (API, features, forecasting) should import
instead of writing raw SQL against tables they're guessing at.
"""
from sqlalchemy import (
    Column, BigInteger, String, Boolean, Float, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Household(Base):
    __tablename__ = "households"

    household_id = Column(BigInteger, primary_key=True)
    source = Column(String(50), nullable=False)
    dwelling_type = Column(String(100))
    climate_zone = Column(String(100))
    electricity_use_group = Column(String(50))
    has_aircon = Column(Boolean)
    has_generation = Column(Boolean)


class EnergyReading(Base):
    __tablename__ = "energy_readings"

    id = Column(BigInteger, primary_key=True)
    household_id = Column(BigInteger, ForeignKey("households.household_id"))
    reading_timestamp = Column(DateTime, nullable=False)
    consumption_kwh = Column(Float)
    controlled_load_kwh = Column(Float)
    gross_generation_kwh = Column(Float)
    net_generation_kwh = Column(Float)
    other_kwh = Column(Float)
    source = Column(String(50), nullable=False)


class ApplianceReading(Base):
    __tablename__ = "appliance_readings"

    id = Column(BigInteger, primary_key=True)
    household_id = Column(BigInteger, ForeignKey("households.household_id"))
    reading_timestamp = Column(DateTime, nullable=False)
    appliance_name = Column(String(100), nullable=False)
    power_kw = Column(Float)
    source = Column(String(50), nullable=False)


class WeatherReading(Base):
    __tablename__ = "weather_readings"

    id = Column(BigInteger, primary_key=True)
    household_id = Column(BigInteger, ForeignKey("households.household_id"))
    reading_timestamp = Column(DateTime, nullable=False)
    temperature = Column(Float)
    humidity = Column(Float)
    visibility = Column(Float)
    apparent_temperature = Column(Float)
    pressure = Column(Float)
    wind_speed = Column(Float)
    cloud_cover = Column(Float)
    wind_bearing = Column(Float)
    precip_intensity = Column(Float)
    dew_point = Column(Float)
    precip_probability = Column(Float)
    weather_icon = Column(String(100))
    weather_summary = Column(String(255))
    source = Column(String(50), nullable=False)

class Tariff(Base):
    __tablename__ = "tariffs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    household_id = Column(BigInteger, ForeignKey("households.household_id"), nullable=False, index=True)
    peak_rate = Column(Float, nullable=False)       # $/kWh
    offpeak_rate = Column(Float, nullable=False)    # $/kWh