from typing import Any
import os
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("ai-model")

# Constants - OpenRouter API Configuration
API_BASE = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")  # Load from .env file

# Model Selection - Choose one:
# FREE VERSION (no credits needed):
MODEL_NAME = "qwen/qwen3-8b:free"  # Qwen3 8B - FREE

# PAID VERSION (better performance, requires credits):
#MODEL_NAME = "qwen/qwen3-8b"  # Qwen3 8B - Standard (paid)

# Other Qwen3 options:
# MODEL_NAME = "qwen/qwen3-14b"  # Qwen3 14B - More capable
# MODEL_NAME = "qwen/qwen3-32b"  # Qwen3 32B - Most capable

# Validate API key is loaded
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please check your .env file.")

# Helper Functions
async def make_ai_request(
    endpoint: str,
    payload: dict[str, Any],
    method: str = "POST"
) -> dict[str, Any] | None:
    """Make a request to the AI model API with proper error handling."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/mcp-qwen",  # Optional: Your app URL
        "X-Title": "MCP Qwen Server"  # Optional: Your app name
    }
    
    url = f"{API_BASE}/{endpoint}"
    
    async with httpx.AsyncClient() as client:
        try:
            if method == "POST":
                response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            else:
                response = await client.get(url, headers=headers, timeout=30.0)
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            print(f"HTTP Error {e.response.status_code}: {error_detail}")
            return {
                "error": f"HTTP {e.response.status_code}",
                "detail": error_detail,
                "model": payload.get("model") if method == "POST" else "N/A"
            }
        except Exception as e:
            print(f"Request failed: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}

def format_response(data: dict[str, Any]) -> str:
    """Format the AI model's response into a readable string."""
    if "error" in data:
        return f"Error: {data['error']}\nDetails: {data.get('detail', 'No additional details')}"
    
    # Adjust this based on your AI model's response format
    # Example for OpenAI-style responses:
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    
    # Example for other formats:
    if "response" in data:
        return data["response"]
    
    return str(data)

# ============================================================================
# ACTIVE TOOLS - These are currently enabled
# ============================================================================

@mcp.tool()
async def qwen_chat(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """Chat with Qwen 2.5 7B model.

    Args:
        prompt: The user's question or prompt
        system_message: System message to set model behavior (default: "You are a helpful assistant.")
        temperature: Sampling temperature 0-1, higher = more creative (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 1000)
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    data = await make_ai_request("chat/completions", payload)
    
    if not data:
        return "Unable to get response from Qwen."
    
    return format_response(data)


# ============================================================================
# SPECIALIZED TOOLS - Uncomment these when you want to add them
# ============================================================================

# @mcp.tool()
# async def qwen_code_review(
#     code: str,
#     language: str = "python"
# ) -> str:
#     """Review code and provide feedback using Qwen.
#
#     Args:
#         code: The code to review
#         language: Programming language (default: "python")
#     """
#     system_msg = "You are an expert code reviewer. Provide constructive feedback on code quality, potential bugs, and improvements."
#     prompt = f"Review this {language} code:\n\n```{language}\n{code}\n```"
#     
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": system_msg},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.3,  # Lower temperature for more focused analysis
#         "max_tokens": 1500
#     }
#     
#     data = await make_ai_request("chat/completions", payload)
#     return format_response(data) if data else "Unable to review code."


# @mcp.tool()
# async def qwen_translate(
#     text: str,
#     target_language: str,
#     source_language: str = "auto-detect"
# ) -> str:
#     """Translate text using Qwen.
#
#     Args:
#         text: Text to translate
#         target_language: Target language (e.g., "Spanish", "French", "Chinese")
#         source_language: Source language (default: "auto-detect")
#     """
#     if source_language == "auto-detect":
#         prompt = f"Translate the following text to {target_language}:\n\n{text}"
#     else:
#         prompt = f"Translate the following text from {source_language} to {target_language}:\n\n{text}"
#     
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": "You are an expert translator. Provide accurate translations."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.3,
#         "max_tokens": 2000
#     }
#     
#     data = await make_ai_request("chat/completions", payload)
#     return format_response(data) if data else "Unable to translate."


# @mcp.tool()
# async def qwen_summarize(
#     text: str,
#     length: str = "medium",
#     style: str = "bullet-points"
# ) -> str:
#     """Summarize text using Qwen.
#
#     Args:
#         text: Text to summarize
#         length: Summary length: "brief" (2-3 sentences), "medium" (paragraph), "detailed" (multiple paragraphs)
#         style: Output style: "bullet-points", "paragraph", "executive-summary"
#     """
#     length_instructions = {
#         "brief": "in 2-3 sentences",
#         "medium": "in one paragraph",
#         "detailed": "in multiple paragraphs with key details"
#     }
#     
#     style_instructions = {
#         "bullet-points": "Use bullet points to organize the summary.",
#         "paragraph": "Write in paragraph form.",
#         "executive-summary": "Write as an executive summary with clear sections."
#     }
#     
#     prompt = f"Summarize the following text {length_instructions.get(length, 'concisely')}. {style_instructions.get(style, '')}:\n\n{text}"
#     
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": "You are an expert at summarizing content clearly and accurately."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.4,
#         "max_tokens": 1000
#     }
#     
#     data = await make_ai_request("chat/completions", payload)
#     return format_response(data) if data else "Unable to summarize."


# @mcp.tool()
# async def qwen_analyze_sentiment(
#     text: str,
#     detailed: bool = False
# ) -> str:
#     """Analyze the sentiment of text using Qwen.
#
#     Args:
#         text: Text to analyze
#         detailed: If True, provide detailed analysis; if False, brief summary (default: False)
#     """
#     if detailed:
#         prompt = f"Provide a detailed sentiment analysis of this text, including overall sentiment, emotional tone, and specific aspects:\n\n{text}"
#         max_tokens = 800
#     else:
#         prompt = f"Analyze the sentiment of this text (positive/negative/neutral/mixed) and explain briefly:\n\n{text}"
#         max_tokens = 300
#     
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": "You are an expert at analyzing sentiment and emotional tone in text."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.3,
#         "max_tokens": max_tokens
#     }
#     
#     data = await make_ai_request("chat/completions", payload)
#     return format_response(data) if data else "Unable to analyze sentiment."


# @mcp.tool()
# async def qwen_generate_code(
#     description: str,
#     language: str = "python",
#     include_comments: bool = True
# ) -> str:
#     """Generate code using Qwen.
#
#     Args:
#         description: Description of what the code should do
#         language: Programming language (default: "python")
#         include_comments: Include explanatory comments (default: True)
#     """
#     comment_instruction = "Include helpful comments." if include_comments else "Minimize comments."
#     prompt = f"Write {language} code for the following: {description}\n\n{comment_instruction}"
#     
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": f"You are an expert {language} programmer. Write clean, efficient, well-structured code."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.4,
#         "max_tokens": 2000
#     }
#     
#     data = await make_ai_request("chat/completions", payload)
#     return format_response(data) if data else "Unable to generate code."

# Server Runner
def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()