from langchain.tools import BaseTool
from datetime import datetime
import asyncio  # Add this import

class GetCurrentTimeTool(BaseTool):
    name: str = "get_current_time"
    description: str = "Returns the current date and time. Useful for time comparisons or scheduling."

    def _run(self, *args, **kwargs) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    async def _arun(self, *args, **kwargs) -> str:  # New async method
        # Run sync code in async context (safe for quick ops)
        return await asyncio.to_thread(self._run, *args, **kwargs)