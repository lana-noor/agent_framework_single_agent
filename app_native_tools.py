import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict
import os
from dotenv import load_dotenv
from agent_framework import AgentThread, ChatMessageStore
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
import json

# Import your existing tool functions
import sys
sys.path.append('tools')

from tools.ai_search import search_documents  #  AI Search function
from tools.tavily_search import web_search      #  Tavily function

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_PROMPT = os.getenv("AZURE_OPENAI_PROMPT")

# FastAPI app
app = FastAPI(title="Agent Framework Chat API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active threads
active_threads: Dict[str, AgentThread] = {}

# Create the agent with native tools at startup
chat_client = AzureOpenAIChatClient(
    credential=AzureCliCredential(),
    endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION
)

agent = chat_client.create_agent(
    instructions=AZURE_OPENAI_PROMPT,
    tools=[search_documents, web_search]  # Native Python functions
)

def create_message_store():
    """Factory function to create message store."""
    return ChatMessageStore()

def get_or_create_thread(thread_id: str = None):
    """Get existing thread or create new one."""
    if thread_id and thread_id in active_threads:
        return active_threads[thread_id], thread_id
    
    new_thread = AgentThread(message_store=create_message_store())
    new_thread_id = thread_id or f"thread_{os.urandom(8).hex()}"
    active_threads[new_thread_id] = new_thread
    
    return new_thread, new_thread_id

# REST API - Non-streaming
@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Non-streaming endpoint with native tools."""
    try:
        data = await request.json()
        message = data.get("message")
        thread_id = data.get("thread_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        thread, thread_id = get_or_create_thread(thread_id)
        
        # Run agent (tools already attached at agent creation)
        result = await agent.run(message, thread=thread)
        
        return {
            "response": result.text if hasattr(result, 'text') else str(result),
            "thread_id": thread_id
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# REST API with SSE streaming
@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: Request):
    """Streaming endpoint with native tools."""
    try:
        data = await request.json()
        message = data.get("message")
        thread_id = data.get("thread_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        thread, thread_id = get_or_create_thread(thread_id)
        
        async def event_generator():
            try:
                yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id})}\n\n"
                
                # Stream the response (tools already attached at agent creation)
                async for chunk in agent.run_stream(message, thread=thread):
                    if chunk.text:
                        yield f"data: {json.dumps({'type': 'stream', 'content': chunk.text, 'thread_id': thread_id})}\n\n"
                
                yield f"data: {json.dumps({'type': 'end', 'thread_id': thread_id})}\n\n"
                
            except Exception as e:
                print(f"Streaming error: {str(e)}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Thread management endpoints
@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str):
    """Get conversation history."""
    if thread_id not in active_threads:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    thread = active_threads[thread_id]
    message_list = []
    
    if thread.message_store:
        messages = await thread.message_store.list_messages()
        if messages:
            for msg in messages:
                message_list.append({
                    "role": msg.role if hasattr(msg, 'role') else 'unknown',
                    "content": msg.content if hasattr(msg, 'content') else str(msg)
                })
    
    return {
        "thread_id": thread_id,
        "messages": message_list,
        "message_count": len(message_list)
    }

@app.delete("/api/thread/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread."""
    if thread_id not in active_threads:
        raise HTTPException(status_code=404, detail="Thread not found")
    del active_threads[thread_id]
    return {"message": "Thread deleted", "thread_id": thread_id}

@app.get("/api/threads")
async def list_threads():
    """List all active threads."""
    thread_info = []
    for tid, thread in active_threads.items():
        msg_count = 0
        if thread.message_store:
            messages = await thread.message_store.list_messages()
            msg_count = len(messages or [])
        thread_info.append({"thread_id": tid, "message_count": msg_count})
    return {"threads": thread_info, "total": len(thread_info)}

@app.post("/api/clear")
async def clear_all():
    """Clear all threads."""
    count = len(active_threads)
    active_threads.clear()
    return {"message": f"Cleared {count} threads"}

@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "active_threads": len(active_threads),
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "deployment": AZURE_OPENAI_DEPLOYMENT,
        "tools": ["search_documents", "web_search"]
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Agent Framework Chat API")
    print("=" * 60)
    print(f"Azure OpenAI: {AZURE_OPENAI_ENDPOINT}")
    print(f"Deployment: {AZURE_OPENAI_DEPLOYMENT}")
    print("\nNative Tools:")
    print("  - AI Search (search_documents)")
    print("  - Tavily Web Search (web_search)")
    print("=" * 60)
    print("Endpoints:")
    print("  POST /api/chat - Non-streaming")
    print("  POST /api/chat/stream - SSE streaming")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)