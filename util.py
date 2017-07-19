from pathlib import Path
from time import strftime


def timestamp() -> str:
    return strftime('%Y%m%d-%H%M%S')

def create_timestamp_directory():
    p = Path(".") / "plots" / timestamp()

    p.mkdir(exist_ok=True)

    return p

timestamp_directory = create_timestamp_directory()