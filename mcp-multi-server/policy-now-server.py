from typing import Any
import os
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Initialize FastMCP server with unique name
mcp = FastMCP("policy-now-bot")

# Constants - OpenRouter API Configuration
# Replace w/ policy now data
#find out what type of model / system its using
API_BASE = "https://api.onspring.com/" #may not need / replace
API_KEY = os.getenv("POLICYNOW_API_KEY")
#MODEL_NAME = "qwen/qwen3-8b:free" #make policy now or something

# Validate API key is loaded
if not API_KEY:
    raise ValueError("POLICY_API_KEY not found in environment variables. Please check your .env file.")

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
        "HTTP-Referer": "https://github.com/yourusername/mcp-policy-bot",
        "X-Title": "MCP Policy Bot"
    }
    
    url = f"{API_BASE}/{endpoint}"
    
    async with httpx.AsyncClient() as client:
        try:
            if method == "POST":
                response = await client.post(url, headers = headers, json = payload, timeout = 60.0)
            else:
                response = await client.get(url, headers = headers, timeout = 30.0)
            
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

#may be done auto in other instances (by final model)
def format_response(data: dict[str, Any]) -> str:
    """Format the AI model's response into a readable string."""
    if "error" in data:
        return f"Error: {data['error']}\nDetails: {data.get('detail', 'No additional details')}"
    
    # OpenRouter uses OpenAI-compatible response format
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    
    if "response" in data:
        return data["response"]
    
    return str(data)

@mcp.tool()
async def ask_policy_bot(question: str) -> str:
    """Ask about company policies, rules, procedures, and regulations.
    
    Use this tool when users ask about:
    - Company policies and guidelines
    - Rules and regulations
    - Standard procedures
    - Compliance requirements
    - General company standards
    
    Args:
        question: Question about company policy or procedures
    """
    #update / make specific prompt for policy now bot (lots of this is alr known/handled im sure)
    system_prompt = """You are a company policy expert assistant. Your role is to answer questions about:
    - Company policies and guidelines
    - Rules and regulations
    - Standard operating procedures
    - Compliance requirements
    - General company standards
    
    IMPORTANT BOUNDARIES:
    - If asked about PTO balances, vacation requests, or time-off specific questions, respond: 
      "Please use the PTO bot for questions about paid time off and vacation balances."
    - If asked about personal employee information, respond:
      "I can only help with general policy questions. For personal information, please contact HR directly."
    
    Provide clear, helpful answers based on standard workplace policies. If you don't know a specific policy, 
    suggest the user contact their HR department for the most accurate information."""
    
    #update payload to reflect what policy now bot wants
    payload = {
       # "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,  # Lower temperature for more consistent policy responses
        "max_tokens": 1000
    }
    
    #make request (may need to update params to match policyNow bot needs)
    data = await make_ai_request("chat/completions", payload)
    
    if not data:
        return "Unable to get response from PolicyNow Bot."
    
    return format_response(data)

# Run MCP Server
def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()