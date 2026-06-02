"""
Database connection and schema definitions.
Uses SQLAlchemy 2.0 with async-friendly session management.
"""

from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime,
    ForeignKey, Text, create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from contextlib import contextmanager
import structlog

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class Region(Base):
    __tablename__ = "regions"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    country = Column(String(100))
    orders = relationship("Order", back_populates="region")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100))
    unit_price = Column(Float, nullable=False)
    cost_price = Column(Float, nullable=False)
    order_items = relationship("OrderItem", back_populates="product")


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200))
    segment = Column(String(50))  # Enterprise, SMB, Consumer
    orders = relationship("Order", back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    region_id = Column(Integer, ForeignKey("regions.id"))
    order_date = Column(Date, nullable=False)
    status = Column(String(50))  # completed, refunded, pending

    customer = relationship("Customer", back_populates="orders")
    region = relationship("Region", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)  # price at time of order

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


# ── Connection Management ─────────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def init_db(database_url: str):
    """Initialize database engine and session factory."""
    global _engine, _SessionLocal
    _engine = create_engine(
        database_url,
        pool_pre_ping=True,      # detect stale connections
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    logger.info("Database engine initialized")
    return _engine


def get_engine():
    if _engine is None:
        raise RuntimeError("Call init_db() before using the database.")
    return _engine


@contextmanager
def get_session():
    """Context manager for safe session lifecycle."""
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_tables():
    """Create all tables (idempotent)."""
    Base.metadata.create_all(get_engine())
    logger.info("All tables created/verified")


def get_schema_description() -> str:
    """
    Returns a human-readable schema description for the SQL Generator agent.
    Keeping this as a string (not dynamic introspection) so the LLM gets
    a clean, stable context.
    """
    return """
DATABASE SCHEMA — bi_analyst
=============================================================

TABLE: orders
  id          INTEGER  PRIMARY KEY
  customer_id INTEGER  FK → customers.id
  region_id   INTEGER  FK → regions.id
  order_date  DATE     (format: YYYY-MM-DD)
  status      TEXT     values: 'completed', 'refunded', 'pending'

TABLE: order_items
  id         INTEGER PRIMARY KEY
  order_id   INTEGER FK → orders.id
  product_id INTEGER FK → products.id
  quantity   INTEGER
  unit_price FLOAT   (price at time of sale)

TABLE: products
  id         INTEGER PRIMARY KEY
  name       TEXT
  category   TEXT    values: 'Electronics', 'Furniture', 'Software', 'Services', 'Accessories'
  unit_price FLOAT   (current list price)
  cost_price FLOAT   (COGS)

TABLE: customers
  id      INTEGER PRIMARY KEY
  name    TEXT
  email   TEXT
  segment TEXT    values: 'Enterprise', 'SMB', 'Consumer'

TABLE: regions
  id      INTEGER PRIMARY KEY
  name    TEXT    values: 'North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East'
  country TEXT

USEFUL DERIVED COLUMNS:
  revenue        = order_items.quantity * order_items.unit_price
  profit         = (order_items.unit_price - products.cost_price) * order_items.quantity
  profit_margin  = (unit_price - cost_price) / unit_price * 100

NOTES:
  - Always filter status = 'completed' unless the question is about refunds/pipeline
  - Use DATE_TRUNC for time grouping (e.g. DATE_TRUNC('month', order_date))
  - Q4 2025 = order_date BETWEEN '2025-10-01' AND '2025-12-31'
  - Q1 2025 = order_date BETWEEN '2025-01-01' AND '2025-03-31'
=============================================================
"""