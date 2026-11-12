import asyncio
from langchain_openai import ChatOpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_CHAT_MODEL

async def test():
    llm = ChatOpenAI(
        model=OPENROUTER_CHAT_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )
    
    gen = llm.astream("Hello")
    async for chunk in gen:
        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
            print("Usage:", chunk.usage_metadata)
            
if __name__ == "__main__":
    asyncio.run(test())
