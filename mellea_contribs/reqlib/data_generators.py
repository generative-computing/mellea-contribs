"""Data generation utilities for auto-fixing Python Requirements.

This module provides random data generators used to create dummy data
when auto-fixing missing files and DataFrame columns.
"""

import random
from datetime import datetime
from typing import Any, Callable, Dict

try:
    import pycountry
except ImportError:
    pycountry = None

try:
    import lorem
except ImportError:
    lorem = None


def random_datetime() -> datetime:
    """Generate random datetime between 2000-2024."""
    return datetime.fromtimestamp(
        random.uniform(
            datetime.fromisoformat("2000-01-01T00:00:00").timestamp(),
            datetime.fromisoformat("2024-01-01T00:00:00").timestamp()
        )
    )


def random_year() -> int:
    """Generate random year between 2020-2024."""
    return random.randint(2020, 2024)


def random_month() -> int:
    """Generate random month (1-12)."""
    return random.randint(1, 12)


def random_day() -> int:
    """Generate random day (1-31)."""
    return random.randint(1, 31)


def random_hour() -> int:
    """Generate random hour (0-23)."""
    return random.randint(0, 23)


def random_minute() -> int:
    """Generate random minute (0-59)."""
    return random.randint(0, 59)


def random_second() -> int:
    """Generate random second (0-59)."""
    return random.randint(0, 59)


def random_int() -> int:
    """Generate random integer between 0-10."""
    return random.randint(0, 10)


def random_country() -> str:
    """Generate random country name."""
    if pycountry is None:
        # Fallback if pycountry not available
        return random.choice([
            "United States", "Canada", "United Kingdom", "Germany",
            "France", "Japan", "Australia", "Brazil", "India", "China"
        ])
    return random.choice(list(pycountry.countries)).name


def random_name() -> str:
    """Generate random person name."""
    return random.choice([
        "Masataro", "Jason", "Nathan", "Shun", "Xiaojie", "Zhangfan",
        "Alice", "Bob", "Carol", "David", "Emma", "Frank"
    ])


def lorem_paragraph() -> str:
    """Generate lorem ipsum paragraph."""
    if lorem is None:
        # Fallback if lorem not available
        return (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco."
        )
    return lorem.paragraph()


# Mapping from column names to appropriate generators
COLUMN_GENERATORS: Dict[str, Callable[[], Any]] = {
    "date": random_datetime,
    "year": random_year,
    "month": random_month,
    "day": random_day,
    "hour": random_hour,
    "minute": random_minute,
    "second": random_second,
    "country": random_country,
    "name": random_name,
}


def get_generator_for_column(column_name: str) -> Callable[[], Any]:
    """Get appropriate generator for column name, defaulting to random_int."""
    return COLUMN_GENERATORS.get(column_name.lower(), random_int)


def generate_dummy_data(column_name: str, num_rows: int) -> list[Any]:
    """Generate dummy data for a column."""
    generator = get_generator_for_column(column_name)
    return [generator() for _ in range(num_rows)]