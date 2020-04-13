import asyncio
import itertools


def limited_as_completed(coros, limit):
    # Based on https://github.com/andybalaam/asyncioplus/blob/master/asyncioplus/limited_as_completed.py  # noqa
    futures = [asyncio.create_task(c) for c in itertools.islice(coros, 0, limit)]
    pending = [len(futures)]  # list so that we can modify from first_to_finish

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
                        pending[0] -= 1
                    return f.result()

    while pending[0] > 0:
        yield first_to_finish()
