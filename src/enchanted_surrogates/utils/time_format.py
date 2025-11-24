from datetime import timedelta

def format_sec(seconds):
    td = timedelta(seconds=seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d - {hours:02}:{minutes:02}:{seconds:02}"

