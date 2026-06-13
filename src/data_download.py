
from __future__ import annotations
from pathlib import Path


def main() -> None:
    """The repository ships with data/raw/online_retail_sample.csv.

    I keep this entry point so the workflow looks familiar, but the sample was
    generated locally from documented assumptions instead of downloaded from a
    third-party service. That makes the project reproducible without credentials
    or internet access.
    """
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "online_retail_sample.csv"
    print(f"Sample data already available: {path}")

if __name__ == "__main__":
    main()
