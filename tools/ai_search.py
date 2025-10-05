"""
Azure AI Search Tool for Microsoft Agent Framework
Searches AI Change Readiness Data for insights on employee readiness and leadership strategies.
"""

import os
from typing import Annotated, List
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient  
from azure.search.documents.models import VectorizableTextQuery
from pydantic import Field

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

def search_documents(
    query: Annotated[str, Field(description="The search query for AI Change Readiness insights.")]
) -> str:
    """
    Use for insights on employee readiness, leadership strategies, and best practices for guiding AI transformation.
    Returns a plain-text context string for the calling agent.
    """
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY or not SEARCH_INDEX_NAME:
        return "Error: Azure Search credentials not configured (check AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX)."

    credential = AzureKeyCredential(AZURE_SEARCH_KEY)
    client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=credential,
    )

    try:
        # Semantic hybrid + vector query (requires semantic config + vector fields in your index)
        results = client.search(  # <-- NO await
            search_text=query,
            vector_queries=[
                VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=50,
                    fields="text_vector"
                )
            ],
            query_type="semantic",
            semantic_configuration_name="my-semantic-config",
            search_fields=["text"],
            top=5,
            include_total_count=True,
        )

        retrieved_texts: List[str] = []
        for r in results:  # <-- NO async for
            txt = (r.get("text") or r.get("content") or r.get("chunk") or "").strip()
            if txt:
                retrieved_texts.append(txt)

        if not retrieved_texts:
            return "No relevant documents found for this query."

        return f"Found {len(retrieved_texts)} relevant documents:\n\n" + "\n\n---\n\n".join(retrieved_texts)

    except Exception as e:
        return f"Error performing search: {e}"
    finally:
        client.close()