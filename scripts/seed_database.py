"""
Demo Database Seeder
─────────────────────
Populates bi_analyst database with realistic sales data spanning 2024–2025.
Run once: python scripts/seed_database.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import date, timedelta
from sqlalchemy.orm import Session
from backend.db.connection import init_db, create_tables, get_engine
from backend.db.connection import Base, Region, Product, Customer, Order, OrderItem

# ── Load settings ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/bi_analyst")

random.seed(42)  # reproducible data


# ── Reference Data ─────────────────────────────────────────────────────────────

REGIONS_DATA = [
    ("North America", "United States"),
    ("Europe", "Germany"),
    ("Asia Pacific", "Japan"),
    ("Latin America", "Brazil"),
    ("Middle East", "UAE"),
]

PRODUCTS_DATA = [
    # (name, category, unit_price, cost_price)
    ("Enterprise Suite Pro", "Software", 1200.0, 150.0),
    ("Analytics Dashboard", "Software", 499.0, 60.0),
    ("Cloud Storage 10TB", "Software", 299.0, 40.0),
    ("Laptop Pro M4", "Electronics", 2499.0, 1600.0),
    ("Wireless Headset Pro", "Electronics", 349.0, 120.0),
    ("4K Monitor 32\"", "Electronics", 699.0, 350.0),
    ("Ergonomic Chair", "Furniture", 849.0, 300.0),
    ("Standing Desk XL", "Furniture", 1299.0, 480.0),
    ("USB-C Hub 12-Port", "Accessories", 89.0, 22.0),
    ("Webcam 4K Ultra", "Accessories", 179.0, 55.0),
    ("Implementation Services", "Services", 5000.0, 2000.0),
    ("Support & Maintenance", "Services", 1800.0, 600.0),
]

CUSTOMER_NAMES = [
    ("Apex Technologies", "Enterprise"),
    ("BlueStar Retail", "SMB"),
    ("Cascade Digital", "Enterprise"),
    ("Digi Corp", "SMB"),
    ("EdgeNet Solutions", "Enterprise"),
    ("FusionWare Inc", "SMB"),
    ("GlobalTech Group", "Enterprise"),
    ("Harbor Systems", "SMB"),
    ("InnovateCo", "Consumer"),
    ("JetStream Labs", "Enterprise"),
    ("Keystone Media", "Consumer"),
    ("LiquidLogic", "SMB"),
    ("MegaScale AI", "Enterprise"),
    ("NovaByte", "SMB"),
    ("Orbit Analytics", "Enterprise"),
    ("PinnacleData", "Consumer"),
    ("QuantumLeap Inc", "Enterprise"),
    ("RapidBuild", "SMB"),
    ("Skyline Ventures", "Enterprise"),
    ("TechPulse", "Consumer"),
]

# Regional seasonality multipliers per quarter
REGIONAL_SEASONALITY = {
    "North America": {1: 0.85, 2: 1.0, 3: 1.1, 4: 1.3},
    "Europe":        {1: 0.9,  2: 1.05, 3: 0.8, 4: 1.25},
    "Asia Pacific":  {1: 1.1,  2: 0.95, 3: 1.0, 4: 1.15},
    "Latin America": {1: 0.8,  2: 0.9,  3: 1.0, 4: 0.95},
    "Middle East":   {1: 0.85, 2: 0.9,  3: 0.7, 4: 1.1},
}

# Simulate some underperforming products
UNDERPERFORM_PRODUCTS = {"Ergonomic Chair", "USB-C Hub 12-Port"}


def generate_order_date() -> date:
    """Random date in 2024-2025, biased toward recent months."""
    start = date(2024, 1, 1)
    end = date(2026, 5, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def seed():
    print("🌱 Seeding database...")
    engine = init_db(DB_URL)
    create_tables()

    with Session(engine) as session:
        # Clear existing data
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        print("  ↳ Cleared existing data")

        # Regions
        regions = {name: Region(name=name, country=country) for name, country in REGIONS_DATA}
        session.add_all(regions.values())
        session.flush()
        print(f"  ↳ {len(regions)} regions")

        # Products
        products = {}
        for name, category, unit_price, cost_price in PRODUCTS_DATA:
            p = Product(name=name, category=category, unit_price=unit_price, cost_price=cost_price)
            products[name] = p
            session.add(p)
        session.flush()
        print(f"  ↳ {len(products)} products")

        # Customers
        customers = []
        for name, segment in CUSTOMER_NAMES:
            c = Customer(name=name, email=f"{name.lower().replace(' ', '.')}@example.com", segment=segment)
            customers.append(c)
            session.add(c)
        session.flush()
        print(f"  ↳ {len(customers)} customers")

        # Orders + Items
        order_count = 0
        item_count = 0
        product_list = list(products.values())
        region_list = list(regions.values())

        for _ in range(2000):
            customer = random.choice(customers)
            region = random.choice(region_list)
            order_date = generate_order_date()
            quarter = (order_date.month - 1) // 3 + 1
            status = random.choices(
                ["completed", "refunded", "pending"],
                weights=[88, 7, 5]
            )[0]

            order = Order(
                customer_id=customer.id,
                region_id=region.id,
                order_date=order_date,
                status=status,
            )
            session.add(order)
            session.flush()
            order_count += 1

            # 1–4 items per order
            n_items = random.randint(1, 4)
            chosen_products = random.sample(product_list, min(n_items, len(product_list)))

            for product in chosen_products:
                # Seasonality + regional adjustment
                season = REGIONAL_SEASONALITY.get(region.name, {}).get(quarter, 1.0)

                # Underperforming products sell less
                if product.name in UNDERPERFORM_PRODUCTS:
                    season *= random.uniform(0.3, 0.6)

                quantity = max(1, int(random.randint(1, 8) * season))
                # Slight price variance (discounts/premiums)
                price_factor = random.uniform(0.9, 1.05)
                unit_price = round(product.unit_price * price_factor, 2)

                item = OrderItem(
                    order_id=order.id,
                    product_id=product.id,
                    quantity=quantity,
                    unit_price=unit_price,
                )
                session.add(item)
                item_count += 1

        session.commit()
        print(f"  ↳ {order_count} orders, {item_count} order items")
        print("✅ Database seeded successfully!")
        print("\nSample questions to try:")
        print("  • 'Show me revenue trends by region for Q4 2025'")
        print("  • 'Which products are underperforming this quarter?'")
        print("  • 'Compare sales performance across customer segments'")
        print("  • 'What are the top 5 customers by total revenue?'")
        print("  • 'Forecast next quarter revenue based on trends'")


if __name__ == "__main__":
    seed()