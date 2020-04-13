import asyncio
import functools


def unasync(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        asyncio.create_task(f(*args, **kwargs))

    return wrapper
