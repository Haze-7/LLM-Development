from typing import Any
import os
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server with unique name
mcp = FastMCP("pto-bot") #not secure/ on http rn

# Constants - OpenRouter API Configuration
API_BASE = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "qwen/qwen3-8b:free"

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
        "HTTP-Referer": "https://github.com/yourusername/mcp-pto-bot",
        "X-Title": "MCP PTO Bot"
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
    
    # OpenRouter uses OpenAI-compatible response format
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    
    if "response" in data:
        return data["response"]
    
    return str(data)

# Mock PTO Database (In real system, this would be a real database)
MOCK_PTO_DATA = {
    "EMP001": {"name": "John Doe", "pto_balance": 15.5, "pto_used": 4.5, "pto_total": 20},
    "EMP002": {"name": "Jane Smith", "pto_balance": 22.0, "pto_used": 3.0, "pto_total": 25},
    "EMP003": {"name": "Bob Johnson", "pto_balance": 8.0, "pto_used": 12.0, "pto_total": 20},
}

def get_pto_data(employee_id: str) -> dict | None:
    """Get PTO data for an employee (mock function)"""
    return MOCK_PTO_DATA.get(employee_id)

# Tool: PTO Bot
@mcp.tool()
async def ask_pto_bot(question: str, employee_id: str = None) -> str:
    """Ask about PTO (Paid Time Off), vacation balances, and time-off requests.
    
    Use this tool when users ask about:
    - PTO balance and availability
    - Vacation days remaining
    - Time-off requests and approval
    - Sick leave balance
    - Holiday schedules
    
    Args:
        question: Question about PTO or time off
        employee_id: Optional employee ID (e.g., "EMP001") for personalized balance info
    """
    # Build context with PTO data if employee_id is provided
    pto_context = ""
    if employee_id:
        pto_data = get_pto_data(employee_id)
        if pto_data:
            pto_context = f"""
EMPLOYEE PTO DATA for {employee_id}:
- Name: {pto_data['name']}
- PTO Balance: {pto_data['pto_balance']} days
- PTO Used: {pto_data['pto_used']} days
- Total Annual PTO: {pto_data['pto_total']} days

Use this data to answer the user's question.
"""
        else:
            pto_context = f"\nNote: No PTO data found for employee ID {employee_id}. Provide general PTO information."
    
    system_msg = f"""You are a PTO (Paid Time Off) specialist assistant. Your role is to answer questions about:
    - PTO balances and availability
    - Vacation days and time-off requests
    - Sick leave policies
    - Holiday schedules
    - Time-off approval processes

{pto_context}

IMPORTANT BOUNDARIES:
    - If asked about general company policies (not PTO-related), respond:
      "Please use the Policy bot for general company policy questions."
    - If asked about other HR topics (benefits, payroll, etc.), respond:
      "I specialize in PTO. For other HR questions, please contact the HR department directly."
    
Provide clear, helpful answers about time off. If specific PTO data isn't available, 
suggest the user contact HR or provide their employee ID for personalized information."""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,  # Lower temperature for consistent responses
        "max_tokens": 1000
    }
    
    data = await make_ai_request("chat/completions", payload)
    
    if not data:
        return "Unable to get response from PTO Bot."
    
    return format_response(data)

# Server Runner
def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()