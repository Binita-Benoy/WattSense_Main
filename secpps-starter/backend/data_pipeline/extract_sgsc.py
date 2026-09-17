from pathlib import Path
import argparse
import pandas as pd


# =========================================================
# Configuration / Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

SUMMARY_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "sgsc_customer_summary.csv"
)

METADATA_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "sgsc-ct_customer-household-data-revised.csv"
)

# Stores information about the final 30 selected households.
SELECTION_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "sgsc_selected_30_households.csv"
)

# Existing 6.5 GB SGSC interval dataset.
# We reference the existing file rather than copying it
# into the WattSense repository.
RAW_INTERVAL_FILE = Path(
    "/Users/alka/Documents/UNI/Fourth Semester/Capstone/"
    "data/raw/CD_INTERVAL_READING_ALL_NO_QUOTES-1.csv"
)

# This is the file consumed by clean.py.
OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "sgsc_selected_households.csv"
)


# =========================================================
# Selection Settings
# =========================================================

TARGET_HOUSEHOLDS = 30
MIN_READING_COUNT = 40000


# Keep the original five households so that existing
# development/testing work remains compatible.
ORIGINAL_FIVE = [
    10042974,
    10346126,
    10733884,
    10859314,
    11228008,
]


# =========================================================
# Select 30 Households
# =========================================================

def select_households():
    print("Loading SGSC customer summary...")

    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Customer summary not found:\n{SUMMARY_FILE}"
        )

    summary = pd.read_csv(SUMMARY_FILE)

    print("Loading SGSC household metadata...")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Household metadata not found:\n{METADATA_FILE}"
        )

    metadata = pd.read_csv(
        METADATA_FILE,
        low_memory=False,
    )

    # -----------------------------------------------------
    # Standardise customer IDs
    # -----------------------------------------------------

    summary["CUSTOMER_ID"] = pd.to_numeric(
        summary["CUSTOMER_ID"],
        errors="coerce",
    ).astype("Int64")

    metadata["CUSTOMER_KEY"] = pd.to_numeric(
        metadata["CUSTOMER_KEY"],
        errors="coerce",
    ).astype("Int64")

    # -----------------------------------------------------
    # Keep household characteristics needed by WattSense
    # -----------------------------------------------------

    metadata_columns = [
        "CUSTOMER_KEY",
        "ASSRTD_CLIMATE_ZONE_DESC",
        "DWELLING_TYPE_CD",
        "ASSRTD_ELECTRICITY_USE_GRP_CD",
        "HAS_AIRCON",
        "HAS_GENERATION",
    ]

    metadata = metadata[
        metadata_columns
    ].copy()

    # -----------------------------------------------------
    # Deduplicate metadata
    #
    # SGSC can contain multiple metadata records for the
    # same customer. Prefer the record containing the most
    # complete household information.
    # -----------------------------------------------------

    characteristic_columns = [
        "ASSRTD_CLIMATE_ZONE_DESC",
        "DWELLING_TYPE_CD",
        "ASSRTD_ELECTRICITY_USE_GRP_CD",
        "HAS_AIRCON",
        "HAS_GENERATION",
    ]

    metadata["_COMPLETENESS"] = (
        metadata[characteristic_columns]
        .notna()
        .sum(axis=1)
    )

    metadata = (
        metadata
        .sort_values(
            "_COMPLETENESS",
            ascending=False,
        )
        .drop_duplicates("CUSTOMER_KEY")
        .drop(columns="_COMPLETENESS")
    )

    # -----------------------------------------------------
    # Combine reading-quality information with household
    # characteristics.
    # -----------------------------------------------------

    combined = summary.merge(
        metadata,
        left_on="CUSTOMER_ID",
        right_on="CUSTOMER_KEY",
        how="inner",
    )

    print(
        "Customers with reading summary and metadata: "
        f"{len(combined):,}"
    )

    # -----------------------------------------------------
    # Data-quality filter
    #
    # We want long, clean histories for forecasting.
    # -----------------------------------------------------

    eligible = combined[
        (
            combined["MISSING_ENERGY_PERCENT"]
            == 0
        )
        & (
            combined["READING_COUNT"]
            >= MIN_READING_COUNT
        )
    ].copy()

    print(
        "High-quality eligible households: "
        f"{len(eligible):,}"
    )

    if len(eligible) < TARGET_HOUSEHOLDS:
        raise RuntimeError(
            f"Only {len(eligible)} households meet the "
            f"quality criteria, but "
            f"{TARGET_HOUSEHOLDS} are required."
        )

    # -----------------------------------------------------
    # Calculate metadata completeness
    # -----------------------------------------------------

    eligible["METADATA_COMPLETENESS"] = (
        eligible[characteristic_columns]
        .notna()
        .sum(axis=1)
    )

    # Prefer:
    # 1. complete metadata
    # 2. larger reading count
    eligible = eligible.sort_values(
        [
            "METADATA_COMPLETENESS",
            "READING_COUNT",
        ],
        ascending=[
            False,
            False,
        ],
    )

    # -----------------------------------------------------
    # Start with original five
    # -----------------------------------------------------

    selected_ids = []

    eligible_ids = set(
        eligible["CUSTOMER_ID"]
        .dropna()
        .astype(int)
        .tolist()
    )

    for customer_id in ORIGINAL_FIVE:
        selected_ids.append(customer_id)

        if customer_id not in eligible_ids:
            print(
                "WARNING: original household "
                f"{customer_id} does not meet the "
                "current quality filter, but it is being "
                "retained for compatibility."
            )

    # -----------------------------------------------------
    # Remaining candidate households
    # -----------------------------------------------------

    remaining = eligible[
        ~eligible["CUSTOMER_ID"].isin(
            selected_ids
        )
    ].copy()

    # Prefer households with known dwelling type.
    #
    # This improves metadata quality without changing
    # the database schema or relaxing energy-data quality.
    remaining["_DWELLING_MISSING"] = (
        remaining["DWELLING_TYPE_CD"]
        .isna()
        .astype(int)
    )

    remaining = remaining.sort_values(
        [
            "_DWELLING_MISSING",
            "METADATA_COMPLETENESS",
            "READING_COUNT",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    # -----------------------------------------------------
    # Balance electricity-use groups
    #
    # SGSC contains:
    # HI
    # MED
    # LOW
    #
    # Use round-robin selection so one consumption group
    # does not dominate the sample.
    # -----------------------------------------------------

    usage_groups = (
        remaining[
            "ASSRTD_ELECTRICITY_USE_GRP_CD"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "Electricity-use groups available: "
        f"{usage_groups}"
    )

    group_positions = {
        group: 0
        for group in usage_groups
    }

    while len(selected_ids) < TARGET_HOUSEHOLDS:

        added = False

        for group in usage_groups:

            candidates = remaining[
                remaining[
                    "ASSRTD_ELECTRICITY_USE_GRP_CD"
                ]
                == group
            ]

            position = group_positions[group]

            while position < len(candidates):

                row = candidates.iloc[position]

                position += 1

                group_positions[group] = position

                customer_id = int(
                    row["CUSTOMER_ID"]
                )

                if customer_id not in selected_ids:

                    selected_ids.append(
                        customer_id
                    )

                    added = True

                    break

            if (
                len(selected_ids)
                >= TARGET_HOUSEHOLDS
            ):
                break

        if not added:
            break

    # -----------------------------------------------------
    # Fallback
    #
    # If balancing categories did not reach 30, take the
    # next highest-quality eligible households.
    # -----------------------------------------------------

    if len(selected_ids) < TARGET_HOUSEHOLDS:

        for customer_id in remaining[
            "CUSTOMER_ID"
        ]:

            customer_id = int(customer_id)

            if customer_id not in selected_ids:
                selected_ids.append(
                    customer_id
                )

            if (
                len(selected_ids)
                == TARGET_HOUSEHOLDS
            ):
                break

    if len(selected_ids) != TARGET_HOUSEHOLDS:
        raise RuntimeError(
            "Could not select exactly "
            f"{TARGET_HOUSEHOLDS} households."
        )

    # -----------------------------------------------------
    # Retrieve selected household metadata
    # -----------------------------------------------------

    selected = combined[
        combined["CUSTOMER_ID"].isin(
            selected_ids
        )
    ].copy()

    # Preserve selection order.
    selection_order = {
        customer_id: index
        for index, customer_id
        in enumerate(selected_ids)
    }

    selected["_ORDER"] = (
        selected["CUSTOMER_ID"]
        .map(selection_order)
    )

    selected = (
        selected
        .sort_values("_ORDER")
        .drop(columns="_ORDER")
    )

    # -----------------------------------------------------
    # Save selection information
    #
    # These fields map directly to information already
    # supported by the existing households table.
    # -----------------------------------------------------

    output_columns = [
        "CUSTOMER_ID",
        "READING_COUNT",
        "FIRST_READING",
        "LAST_READING",
        "MISSING_ENERGY_PERCENT",
        "ASSRTD_ELECTRICITY_USE_GRP_CD",
        "DWELLING_TYPE_CD",
        "ASSRTD_CLIMATE_ZONE_DESC",
        "HAS_AIRCON",
        "HAS_GENERATION",
    ]

    selected[
        output_columns
    ].to_csv(
        SELECTION_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Display selected households
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("SELECTED HOUSEHOLDS")
    print("=" * 60)

    print(
        selected[
            output_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Total selected:",
        len(selected),
    )

    print()
    print(
        "Electricity-use distribution:"
    )

    print(
        selected[
            "ASSRTD_ELECTRICITY_USE_GRP_CD"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Climate-zone distribution:"
    )

    print(
        selected[
            "ASSRTD_CLIMATE_ZONE_DESC"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Dwelling-type distribution:"
    )

    print(
        selected[
            "DWELLING_TYPE_CD"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Air-conditioning distribution:"
    )

    print(
        selected[
            "HAS_AIRCON"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Generation distribution:"
    )

    print(
        selected[
            "HAS_GENERATION"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Selection saved to:"
    )

    print(SELECTION_FILE)

    return selected_ids


# =========================================================
# Extract Interval Data
# =========================================================

def extract_interval_data(selected_ids):

    if not RAW_INTERVAL_FILE.exists():
        raise FileNotFoundError(
            "Raw SGSC interval file not found:\n"
            f"{RAW_INTERVAL_FILE}"
        )

    print()
    print("=" * 60)
    print("EXTRACTING SGSC INTERVAL DATA")
    print("=" * 60)

    print(
        f"Source: {RAW_INTERVAL_FILE}"
    )

    print(
        f"Households: {len(selected_ids)}"
    )

    print(
        "Scanning the full SGSC dataset..."
    )

    selected_chunks = []

    total_rows = 0
    selected_rows = 0

    # Process the large source file in chunks rather than
    # attempting to load 6.5 GB into memory.
    reader = pd.read_csv(
        RAW_INTERVAL_FILE,
        chunksize=500_000,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        total_rows += len(chunk)

        chunk["CUSTOMER_ID"] = pd.to_numeric(
            chunk["CUSTOMER_ID"],
            errors="coerce",
        )

        matched = chunk[
            chunk["CUSTOMER_ID"].isin(
                selected_ids
            )
        ].copy()

        if not matched.empty:

            selected_chunks.append(
                matched
            )

            selected_rows += len(
                matched
            )

        # Progress every 5 million rows.
        if chunk_number % 10 == 0:

            print(
                f"Processed "
                f"{total_rows:,} rows "
                f"| selected "
                f"{selected_rows:,}"
            )

    if not selected_chunks:
        raise RuntimeError(
            "No interval readings were found "
            "for the selected households."
        )

    result = pd.concat(
        selected_chunks,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Validate extraction
    # -----------------------------------------------------

    extracted_ids = set(
        result["CUSTOMER_ID"]
        .dropna()
        .astype(int)
        .unique()
    )

    expected_ids = set(
        selected_ids
    )

    missing_households = (
        expected_ids
        - extracted_ids
    )

    if missing_households:
        raise RuntimeError(
            "The following selected households "
            "were not found in the raw dataset: "
            f"{sorted(missing_households)}"
        )

    if len(extracted_ids) != TARGET_HOUSEHOLDS:
        raise RuntimeError(
            "Expected "
            f"{TARGET_HOUSEHOLDS} households "
            "but extracted "
            f"{len(extracted_ids)}."
        )

    # -----------------------------------------------------
    # Save extracted interval readings
    # -----------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Rows scanned: "
        f"{total_rows:,}"
    )

    print(
        f"Rows selected: "
        f"{len(result):,}"
    )

    print(
        "Households extracted: "
        f"{len(extracted_ids)}"
    )

    print()
    print(
        "Rows per household:"
    )

    print(
        result[
            "CUSTOMER_ID"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Extracted data saved to:"
    )

    print(OUTPUT_FILE)


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Select 30 high-quality SGSC households "
            "and optionally extract their interval readings."
        )
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help=(
            "After selecting the households, scan the "
            "full SGSC interval dataset and extract "
            "their readings."
        ),
    )

    args = parser.parse_args()

    selected_ids = select_households()

    if args.extract:

        extract_interval_data(
            selected_ids
        )

    else:

        print()
        print(
            "Selection only. "
            "Raw interval data was NOT extracted."
        )

        print(
            "Review the selected households first."
        )

        print(
            "When ready, run:"
        )

        print(
            "python "
            "backend/data_pipeline/extract_sgsc.py "
            "--extract"
        )


if __name__ == "__main__":
    main()