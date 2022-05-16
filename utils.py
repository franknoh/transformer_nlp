def sec_to_time(sec):
    min = sec // 60
    sec %= 60

    hr = min // 60
    min %= 60

    day = hr // 24
    hr %= 24

    if day > 0:
        return f"{day}d {hr}h {min}m {sec}s"
    elif hr > 0:
        return f"{hr}h {min}m {sec}s"
    elif min > 0:
        return f"{min}m {sec}s"
    else:
        return f"{sec}s"


__all__ = ["sec_to_time"]
