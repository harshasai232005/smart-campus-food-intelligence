import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


load_dotenv()


def get_engine():

    host = os.getenv(
        "DB_HOST",
        "localhost"
    )

    port = int(
        os.getenv(
            "DB_PORT",
            "3306"
        )
    )

    database = os.getenv(
        "DB_NAME",
        "campus_food_intelligence"
    )

    username = os.getenv(
        "DB_USER",
        "root"
    )

    password = os.getenv(
        "DB_PASSWORD",
        ""
    )

    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database
    )

    return create_engine(
        connection_url,
        pool_pre_ping=True
    )


def test_connection():

    engine = get_engine()

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        value = result.scalar()

        print(
            "Database test result:",
            value
        )