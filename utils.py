import json
import os

CWD = os.getcwd()
DATABASE_PATH = os.path.join(CWD, "database.json")
TEMP_PATH = os.path.join(CWD, "temp")
OUTPUT_PATH = os.path.join(CWD, "result")
MEL_STEP_SIZE = 16

os.makedirs(TEMP_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_database() -> dict:
    """Load the database."""
    print("Loading database")
    if not os.path.exists(DATABASE_PATH):
        print("Creating database")
        with open(DATABASE_PATH, "w+") as file:
            json.dump({}, file)
            database = {}
    else:
        with open(DATABASE_PATH, "r") as file:
            database = json.load(file)
    return database or {}


def dump_database(database: dict) -> None:
    """Dump the database."""
    print("Dumping database")
    with open(DATABASE_PATH, "w") as file:
        print("Saving database")
        json.dump(database, file, indent=4)


def save_config(
    run_id: str,
    data: dict,
) -> None:
    """Save the configuration to the database."""
    # create the database if it doesn't exist
    print("Saving configuration to database")
    database = load_database()
    database[run_id] = data
    dump_database(database)