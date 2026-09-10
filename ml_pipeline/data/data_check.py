# share_data_summary.py
import pandas as pd
from sqlalchemy import create_engine, text

# Update password
engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/smart_energy"
)

with engine.connect() as conn:

    # All tables
    tables = conn.execute(text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)).fetchall()
    print("TABLES:", [t[0] for t in tables])

    # For each table print schema and sample
    for table in [t[0] for t in tables
                  if "seq" not in t[0]]:
        print(f"\n{'='*60}")
        print(f"TABLE: {table}")
        print(f"{'='*60}")

        # Column info
        cols = conn.execute(text(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}'
        """)).fetchall()
        print("Columns:")
        for col in cols:
            print(f"  {col[0]:<30} {col[1]}")

        # Row count
        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar()
        print(f"Row count: {count:,}")

        # Sample data
        sample = pd.read_sql(
            f"SELECT * FROM {table} LIMIT 5",
            engine
        )
        print("Sample:")
        print(sample.to_string())