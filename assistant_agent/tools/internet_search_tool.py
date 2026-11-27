"""A small DDGS adapter with limits on waiting and outstanding work."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import BoundedSemaphore

from ddgs import DDGS
from ddgs.exceptions import DDGSException


class SearchUnavailable(RuntimeError):
    """The search provider failed or could not finish within the time limit."""


class SearchTool:
    """Search for snippets without blocking the graph past its lookup timeout."""

    def __init__(self):
        """Reserve two worker slots for provider calls that may outlive their wait."""
        self._workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="signal-search")
        self._slots = BoundedSemaphore(2)

    @staticmethod
    def _lookup(query: str) -> list[dict]:
        """Run the provider request and normalize its no-results exception."""
        try:
            with DDGS(timeout=5) as ddgs:
                return ddgs.text(query, max_results=4)
        except DDGSException as error:
            if str(error) == "No results found.":
                return []
            raise SearchUnavailable("Search provider unavailable") from error

    def search(self, query: str, timeout: float) -> list[dict]:
        """Allow at most two outstanding lookups, even if timed-out work is still running."""
        if timeout <= 0 or not self._slots.acquire(blocking=False):
            raise SearchUnavailable("Search capacity or time budget exhausted")
        try:
            future = self._workers.submit(self._lookup, query)
        except RuntimeError as error:
            self._slots.release()
            raise SearchUnavailable("Search is shutting down") from error
        future.add_done_callback(lambda _: self._slots.release())
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as error:
            future.cancel()
            raise SearchUnavailable("Search timed out") from error
        except (OSError, DDGSException) as error:
            raise SearchUnavailable("Search provider unavailable") from error

    def close(self) -> None:
        """Stop accepting lookups and cancel any work that has not started."""
        self._workers.shutdown(wait=False, cancel_futures=True)
