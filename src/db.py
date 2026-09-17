import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# Load .env and allow it to override old environment variables
load_dotenv(override=True)


def get_engine():
    """
    Create a SQLAlchemy MySQL engine.

    For external Railway connections, use MYSQL_PUBLIC_URL.
    """

    public_url = os.getenv(
        "MYSQL_PUBLIC_URL",
        ""
    ).strip()

    if public_url:

        # Railway provides mysql://...
        # SQLAlchemy + PyMySQL uses mysql+pymysql://...
        if public_url.startswith("mysql://"):
            public_url = public_url.replace(
                "mysql://",
                "mysql+pymysql://",
                1
            )

        return create_engine(
            public_url,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 15
            }
        )

    raise RuntimeError(
        "MYSQL_PUBLIC_URL is not set in .env"
    )


def test_connection():

    engine = get_engine()

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        print(
            "Database test result:",
            result.scalar()
        )