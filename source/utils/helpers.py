from humanfriendly import format_timespan

def to_hms(seconds: int | float) -> str:
    return format_timespan(seconds)