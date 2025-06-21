import asyncio
import logging
import random


def retry_async(retries=3, delay=5, backoff=2):
    """
    A decorator for retrying an async function with exponential backoff.
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            m_retries, m_delay = retries, delay
            while m_retries > 1:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logging.warning(
                        f"'{func.__name__}' failed. Retrying in {m_delay} seconds... Error: {e}"
                    )
                    await asyncio.sleep(m_delay + random.uniform(0, 1))  # Add jitter
                    m_retries -= 1
                    m_delay *= backoff
            # Final attempt
            return await func(*args, **kwargs)

        return wrapper

    return decorator
