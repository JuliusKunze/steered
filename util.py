from time import strftime


def timestamp() -> str:
    return strftime('%Y%m%d-%H%M%S')
