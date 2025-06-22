import asyncio
import logging
import random


def retry_async(
    retries=3,
    delay=5,
    backoff=2,
):
    """
    A decorator for retrying an async function with exponential backoff.
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            m_retries, m_delay = retries, delay
            class_name = (
                args[0].__class__.__name__
                if args and hasattr(args[0], "__class__")
                else ""
            )

            while m_retries > 1:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logging.error(
                        f"\n[{class_name}] '{func.__name__}' failed. Retries left: {m_retries-1}, Args: {args}, Kwargs: {kwargs}. Retrying in {m_delay} seconds... Error: {e}"
                    )
                    await asyncio.sleep(m_delay + random.uniform(0, 1))
                    m_retries -= 1
                    m_delay *= backoff

            return await func(*args, **kwargs)

        return wrapper

    return decorator
