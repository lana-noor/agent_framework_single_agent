import os
import json
from typing import Annotated
from dotenv import load_dotenv
from tavily import TavilyClient
from pydantic import Field

load_dotenv()

async def web_search(
    query: Annotated[str, Field(description="The search query")]
) -> str:
    """
    Search the web using Tavily and return raw JSON results.
    
    Use when you need current information from the web.
    """
    try:
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return json.dumps({"error": "TAVILY_API_KEY not configured"})
        
        client = TavilyClient(tavily_key)
        result = client.search(
            query=query,
            max_results=8,
            search_depth="advanced",
            include_answer=True,
        )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})