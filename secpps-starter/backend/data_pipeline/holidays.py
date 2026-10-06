from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

NSW_PUBLIC_HOLIDAYS = [
    # 2011
    "2011-12-25",  # Christmas Day
    "2011-12-26",  # Boxing Day
    "2011-12-27",  # Additional public holiday

    # 2012
    "2012-01-01",  # New Year's Day
    "2012-01-02",  # Additional public holiday
    "2012-01-26",  # Australia Day
    "2012-04-06",  # Good Friday
    "2012-04-07",  # Easter Saturday
    "2012-04-08",  # Easter Sunday
    "2012-04-09",  # Easter Monday
    "2012-04-25",  # Anzac Day
    "2012-06-11",  # Queen's Birthday
    "2012-10-01",  # Labour Day
    "2012-12-25",  # Christmas Day
    "2012-12-26",  # Boxing Day

    # 2013
    "2013-01-01",  # New Year's Day

    # Australia Day was Saturday 26 Jan.
    # Under NSW rules the public holiday was Monday 28 Jan,
    # rather than both dates.
    "2013-01-28",  # Australia Day public holiday

    "2013-03-29",  # Good Friday
    "2013-03-30",  # Easter Saturday
    "2013-03-31",  # Easter Sunday
    "2013-04-01",  # Easter Monday
    "2013-04-25",  # Anzac Day
    "2013-06-10",  # Queen's Birthday
    "2013-10-07",  # Labour Day
    "2013-12-25",  # Christmas Day
    "2013-12-26",  # Boxing Day

    # 2014 - dataset ends 3 March
    "2014-01-01",  # New Year's Day

    # Australia Day was Sunday 26 Jan.
    # Monday 27 Jan was the NSW public holiday.
    "2014-01-27",  # Australia Day public holiday
]

def main():
    engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        # Reset first so the script is safely repeatable.
        conn.execute(
            text("""
                UPDATE energy_readings
                SET is_public_holiday = FALSE
                WHERE source = 'SGSC'
            """)
        )

        result = conn.execute(
            text("""
                UPDATE energy_readings
                SET is_public_holiday = TRUE
                WHERE source = 'SGSC'
                  AND reading_timestamp::date = ANY(
                      CAST(:holiday_dates AS date[])
                  )
            """),
            {"holiday_dates": NSW_PUBLIC_HOLIDAYS}
        )

        print(f"Holiday energy rows marked: {result.rowcount:,}")

        verification = conn.execute(
            text("""
                SELECT
                    COUNT(*) AS total_energy_rows,
                    COUNT(*) FILTER (
                        WHERE is_public_holiday = TRUE
                    ) AS public_holiday_rows,
                    COUNT(DISTINCT reading_timestamp::date)
                        FILTER (WHERE is_public_holiday = TRUE)
                        AS public_holiday_dates
                FROM energy_readings
                WHERE source = 'SGSC'
            """)
        ).mappings().one()

        print("\nPublic holiday verification:")
        print(f"Total SGSC energy rows: {verification['total_energy_rows']:,}")
        print(f"Holiday energy rows:    {verification['public_holiday_rows']:,}")
        print(f"Holiday dates present:  {verification['public_holiday_dates']}")


if __name__ == "__main__":
    main()