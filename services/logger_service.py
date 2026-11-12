import logging
import httpx
import asyncio

logger = logging.getLogger(__name__)

JAVA_BACKEND_URL = "http://localhost:8080/api/internal/logs"
INTERNAL_SECRET = "omnitutor-internal-secret-2023"

async def send_log_to_java(level: str, actor: str, action: str, latency_ms: int, token_cost: int):
    """
    Gửi log từ Python sang Java Backend một cách bất đồng bộ để không chặn luồng chính.
    """
    payload = {
        "level": level,
        "actor": actor,
        "action": action,
        "latencyMs": latency_ms,
        "tokenCost": token_cost
    }
    
    headers = {
        "X-Internal-Secret": INTERNAL_SECRET,
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(JAVA_BACKEND_URL, json=payload, headers=headers, timeout=5.0)
            if response.status_code != 200:
                logger.error(f"Lỗi khi gửi log sang Java BE: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Exception khi gửi log sang Java BE: {str(e)}")

def log_event_async(level: str, actor: str, action: str, latency_ms: int, token_cost: int):
    """
    Hàm gọi fire-and-forget để gửi log.
    """
    try:
        # Nếu đang trong event loop của asyncio
        loop = asyncio.get_running_loop()
        loop.create_task(send_log_to_java(level, actor, action, latency_ms, token_cost))
    except RuntimeError:
        # Nếu không có event loop (chạy đồng bộ)
        asyncio.run(send_log_to_java(level, actor, action, latency_ms, token_cost))
