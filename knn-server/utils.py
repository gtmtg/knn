import asyncio
import functools
import itertools


def limited_as_completed(coros, limit):
    # Based on https://github.com/andybalaam/asyncioplus/blob/master/asyncioplus/limited_as_completed.py  # noqa
    futures = [asyncio.create_task(c) for c in itertools.islice(coros, 0, limit)]

    async def first_to_finish():
        while True:
            await asyncio.sleep(0)
            for i, f in enumerate(futures):
                if f is not None and f.done():
                    try:
                        newf = next(coros)
                        futures[i] = asyncio.create_task(newf)
                    except StopIteration:
                        futures[i] = None
                    return f.result()

    while len(futures) > 0:
        yield first_to_finish()


def unasync_eventually(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        asyncio.create_task(f(*args, **kwargs))

    return wrapper


def unasync_now(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper
