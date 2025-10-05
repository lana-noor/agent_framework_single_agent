import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict, List
import os
from dotenv import load_dotenv
from agent_framework import ChatAgent, AgentThread, ChatMessageStore
from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
import json

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_PROMPT = os.getenv("AZURE_OPENAI_PROMPT", "You are a helpful AI assistant.")

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "multimodal-sample-41-index")

# Microsoft Learn MCP URL
MS_LEARN_MCP_URL = "https://learn.microsoft.com/api/mcp"

# FastAPI app
app = FastAPI(title="Agent Framework Chat API - All MCP Tools")

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

def create_message_store():
    """Factory function to create message store."""
    return ChatMessageStore()

def create_chat_client():
    """Create Azure OpenAI Chat Client."""
    if AZURE_OPENAI_API_KEY:
        return AzureOpenAIChatClient(
            api_key=AZURE_OPENAI_API_KEY,
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION
        )
    else:
        return AzureOpenAIChatClient(
            credential=AzureCliCredential(),
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION
        )

async def create_mcp_tools() -> List:
    """Create ALL tools as MCP servers."""
    tools = []
    
    # 1. AI Search MCP (Stdio - Local Python script)
    if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY and AZURE_SEARCH_INDEX:
        ai_search_mcp = MCPStdioTool(
            name="ai-search",
            command="python",
            args=["tools/mcp/ai_search_mcp.py"]
        )
        tools.append(ai_search_mcp)
        print(f"✅ AI Search MCP enabled (index: {AZURE_SEARCH_INDEX})")
    
    # 2. Microsoft Learn MCP (HTTP - Public endpoint, no auth)
    if MS_LEARN_MCP_URL:
        ms_learn_mcp = MCPStreamableHTTPTool(
            name="microsoft-learn",
            url=MS_LEARN_MCP_URL,
        )
        tools.append(ms_learn_mcp)
        print("✅ Microsoft Learn MCP enabled")
    
    # 3. Tavily Web Search MCP (Stdio - Local Python script)
    if os.getenv("TAVILY_API_KEY"):
        tavily_mcp = MCPStdioTool(
            name="tavily-search",
            command="python",
            args=["tools/mcp/tavily_search_mcp.py"]
        )
        tools.append(tavily_mcp)
        print("✅ Tavily Web Search MCP enabled")
    
    return tools

def get_or_create_thread(thread_id: str = None):
    """Get existing thread or create new one."""
    if thread_id and thread_id in active_threads:
        return active_threads[thread_id], thread_id
    
    new_thread = AgentThread(message_store=create_message_store())
    new_thread_id = thread_id or f"thread_{os.urandom(8).hex()}"
    active_threads[new_thread_id] = new_thread
    
    return new_thread, new_thread_id

# REST API with SSE streaming
@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: Request):
    """Streaming endpoint with all MCP tools."""
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
                
                # Get all MCP tools
                mcp_tools = await create_mcp_tools()
                
                # Create agent with MCP tools
                async with ChatAgent(
                    chat_client=create_chat_client(),
                    instructions=AZURE_OPENAI_PROMPT,
                    chat_message_store_factory=create_message_store,
                ) as agent:
                    
                    full_response = ""
                    
                    # Run agent with MCP tools
                    async for chunk in agent.run_stream(message, thread=thread, tools=mcp_tools):
                        if chunk.text:
                            full_response += chunk.text
                            yield f"data: {json.dumps({'type': 'stream', 'content': chunk.text, 'thread_id': thread_id})}\n\n"
                    
                    yield f"data: {json.dumps({'type': 'end', 'thread_id': thread_id, 'full_response': full_response})}\n\n"
                
            except Exception as e:
                print(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# REST API - Non-streaming
@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Non-streaming endpoint with all MCP tools."""
    try:
        data = await request.json()
        message = data.get("message")
        thread_id = data.get("thread_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        thread, thread_id = get_or_create_thread(thread_id)
        
        # Get all MCP tools
        mcp_tools = await create_mcp_tools()
        
        # Create agent
        async with ChatAgent(
            chat_client=create_chat_client(),
            instructions=AZURE_OPENAI_PROMPT,
            chat_message_store_factory=create_message_store,
        ) as agent:
            
            full_response = ""
            
            # Collect all chunks
            async for chunk in agent.run_stream(message, thread=thread, tools=mcp_tools):
                if chunk.text:
                    full_response += chunk.text
            
            return {
                "response": full_response,
                "thread_id": thread_id
            }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket for real-time streaming chat."""
    await websocket.accept()
    thread_id = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            thread_id = message_data.get("thread_id")
            
            thread, thread_id = get_or_create_thread(thread_id)
            
            await websocket.send_json({
                "type": "start",
                "thread_id": thread_id
            })
            
            try:
                # Get all MCP tools
                mcp_tools = await create_mcp_tools()
                
                # Create agent
                async with ChatAgent(
                    chat_client=create_chat_client(),
                    instructions=AZURE_OPENAI_PROMPT,
                    chat_message_store_factory=create_message_store,
                ) as agent:
                    
                    full_response = ""
                    
                    # Stream chunks
                    async for chunk in agent.run_stream(user_message, thread=thread, tools=mcp_tools):
                        if chunk.text:
                            full_response += chunk.text
                            await websocket.send_json({
                                "type": "stream",
                                "content": chunk.text,
                                "thread_id": thread_id
                            })
                    
                    await websocket.send_json({
                        "type": "end",
                        "thread_id": thread_id,
                        "full_response": full_response
                    })
                
            except Exception as e:
                print(f"Streaming error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "thread_id": thread_id
                })
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {thread_id}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

# Get conversation history
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

@app.get("/health")
async def health_check():
    """Health check with MCP status."""
    mcp_tools_enabled = []
    
    if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY:
        mcp_tools_enabled.append(f"AI Search MCP ({AZURE_SEARCH_INDEX})")
    if MS_LEARN_MCP_URL:
        mcp_tools_enabled.append("Microsoft Learn MCP")
    if os.getenv("TAVILY_API_KEY"):
        mcp_tools_enabled.append("Tavily Web Search MCP")
    
    return {
        "status": "healthy",
        "active_threads": len(active_threads),
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "deployment": AZURE_OPENAI_DEPLOYMENT,
        "all_mcp_tools": mcp_tools_enabled
    }

@app.post("/api/clear")
async def clear_all():
    """Clear all threads."""
    count = len(active_threads)
    active_threads.clear()
    return {"message": f"Cleared {count} threads"}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Agent Framework Chat API - ALL MCP Tools")
    print("=" * 60)
    print(f"Azure OpenAI: {AZURE_OPENAI_ENDPOINT}")
    print(f"Deployment: {AZURE_OPENAI_DEPLOYMENT}")
    print("\nAll Tools as MCP Servers:")
    if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY:
        print(f"  ✅ AI Search MCP (Stdio) - Index: {AZURE_SEARCH_INDEX}")
    if MS_LEARN_MCP_URL:
        print("  ✅ Microsoft Learn MCP (HTTP) - No auth")
    if os.getenv("TAVILY_API_KEY"):
        print("  ✅ Tavily Web Search MCP (Stdio)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)