"""
Azure AI Search Tool for Microsoft Agent Framework
Searches AI Change Readiness Data for insights on employee readiness and leadership strategies.
"""

import os
from typing import Annotated
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from pydantic import Field

load_dotenv()

# Azure Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

def search_documents(
    query: Annotated[str, Field(description="Search AI Change Readiness data for insights on employee readiness and leadership strategies.")]
) -> str:
    """
    Search Azure AI Search for AI Change Readiness insights.
    
    Use this tool for insights on employee readiness, leadership strategies, 
    and best practices for guiding AI transformation within organizations.
    
    Args:
        query: The search query string
        
    Returns:
        Context string with relevant search results or error message
    """
    try:
        if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, SEARCH_INDEX_NAME]):
            return "Error: Azure Search credentials not configured."
        
        credential = AzureKeyCredential(AZURE_SEARCH_KEY)
        client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=SEARCH_INDEX_NAME,
            credential=credential,
        )
        
        results = client.search(
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
            top=3,
            include_total_count=True,
        )
        
        retrieved_texts = [result.get("text", "") for result in results if result.get("text")]
        
        if retrieved_texts:
            context_str = "\n\n---\n\n".join(retrieved_texts)
            return f"Found {len(retrieved_texts)} relevant documents:\n\n{context_str}"
        else:
            return "No relevant documents found for this query."
            
    except Exception as e:
        return f"Error performing search: {str(e)}"