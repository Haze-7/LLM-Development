"""
AI Bots MCP Server - Integrated with OnSpring Data Sources

Architecture:
1. OpenRouter provides AI reasoning
2. OnSpring provides enterprise data (with guided discovery)
3. Bots combine AI + real data for intelligent responses

Flow:
- User asks question
- Bot sees catalog (knows what tables exist + example fields)
- Bot decides which table to query
- Schema fetched dynamically for that table only
- Query executes with full context
- AI generates response using real data
"""

from typing import Any, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import os
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("vendor-bots")

# ============================================================================
# AI MODEL PROVIDER (OpenRouter)
# ============================================================================

class ModelProvider(ABC):
    """Abstract base class for AI model providers."""
    
    @abstractmethod
    async def generate_response(self, system_prompt: str, user_message: str) -> str:
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        pass


class OpenRouterProvider(ModelProvider):
    """OpenRouter AI provider."""
    
    def __init__(self, api_key: str, model_name: str = "qwen/qwen3-8b:free"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = "https://openrouter.ai/api/v1"
    
    async def generate_response(self, system_prompt: str, user_message: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/mcp-vendor-bots",
            "X-Title": "MCP Vendor Bots Server"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        url = f"{self.api_base}/chat/completions"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                else:
                    return f"Error: Unexpected response format from {self.model_name}"
                    
            except httpx.HTTPStatusError as e:
                return f"HTTP Error {e.response.status_code}: {e.response.text}"
            except Exception as e:
                return f"Request failed: {str(e)}"
    
    def get_provider_name(self) -> str:
        return f"OpenRouter ({self.model_name})"


# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialize AI Provider
MODEL_PROVIDER = OpenRouterProvider(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="qwen/qwen3-8b:free"
)

# Mock PTO Database (fallback when OnSpring not configured)
MOCK_PTO_DATA = {
    "EMP001": {"name": "John Doe", "pto_balance": 15.5, "pto_used": 4.5, "pto_total": 20},
    "EMP002": {"name": "Jane Smith", "pto_balance": 22.0, "pto_used": 3.0, "pto_total": 25},
    "EMP003": {"name": "Bob Johnson", "pto_balance": 8.0, "pto_used": 12.0, "pto_total": 15},
}


# ============================================================================
# DATA HELPER FUNCTIONS
# ============================================================================

async def get_pto_data(employee_id: str) -> Optional[dict[str, Any]]:
    """Get PTO data -Mock data"""
    return MOCK_PTO_DATA.get(employee_id)


async def get_policy_context() -> str:
    """Get Fake Policy Context"""
    # if not ONSPRING_DATA.enabled:
    #     return ""
    
    #  policy_data = await ONSPRING_DATA.get_policy_data()
    policy_data = {}
    # if not policy_data:
    #     return ""
    
    # Format policy data for prompt
    context = "\n\nRELEVANT POLICY DATA FROM ONSPRING:"
    if isinstance(policy_data, dict) and "items" in policy_data:
        for policy in policy_data["items"][:5]:  # Limit to 5 policies
            # Extract field data
            field_data = {field["fieldId"]: field.get("value", "") 
                         for field in policy.get("fieldData", [])}
            
            record_id = policy.get('recordId')
            # Get some meaningful fields (adjust based on your schema)
            purpose = field_data.get(4785, "")[:100]  # Field 4785 is Purpose
            
            context += f"\n- Policy Record {record_id}: {purpose}..."
    
    return context


# ============================================================================
# TOOL 1: POLICY BOT
# ============================================================================

@mcp.tool()
async def ask_policy_bot(question: str, include_onspring_data: bool = True) -> str:
    """
    Ask about company policies, rules, procedures, and regulations.
    
    This bot uses OnSpring data (when available) to provide accurate,
    company-specific policy information.
    
    Args:
        question: Question about company policy or procedures
        include_onspring_data: Whether to fetch OnSpring data for context
    
    Returns:
        AI-generated answer about company policies
    """
    
    # Get OnSpring catalog context
    catalog_context = ""
    policy_context = ""
    
    # if include_onspring_data and ONSPRING_DATA.enabled:
    #     catalog_context = ONSPRING_DATA.get_catalog_context()
    #     policy_context = await get_policy_context()
    
    system_prompt = f"""You are a company policy expert assistant. Your role is to answer questions about:
- Company policies and guidelines
- Rules and regulations
- Standard operating procedures
- Compliance requirements
- General company standards
{catalog_context}
{policy_context}

IMPORTANT BOUNDARIES:
- If asked about PTO balances, vacation requests, or time-off specific questions, respond: 
  "Please use the PTO bot for questions about paid time off and vacation balances."
- If asked about personal employee information, respond:
  "I can only help with general policy questions. For personal information, please contact HR directly."

Provide clear, helpful answers based on the data provided above."""
    
    try:
        response = await MODEL_PROVIDER.generate_response(
            system_prompt=system_prompt,
            user_message=question
        )
        return response
    except Exception as e:
        return f"Error from Policy Bot: {str(e)}"


# ============================================================================
# TOOL 2: PTO BOT
# ============================================================================

@mcp.tool()
async def ask_pto_bot(question: str, employee_id: Optional[str] = None) -> str:
    """
    Ask about PTO (Paid Time Off), vacation balances, and time-off requests.
    
    This bot fetches real employee data from OnSpring (or mock data) to provide
    personalized PTO information.
    
    Args:
        question: Question about PTO or time off
        employee_id: Optional employee ID for personalized balance info
    
    Returns:
        AI-generated answer about PTO/vacation
    """
    
    # Get OnSpring catalog context
    catalog_context = ""
    # if ONSPRING_DATA.enabled:
    #     catalog_context = ONSPRING_DATA.get_catalog_context()
    
    # Build context with PTO data
    pto_context = ""
    
    if employee_id:
        pto_data = await get_pto_data(employee_id)
        if pto_data:
            pto_context = f"""

EMPLOYEE PTO DATA for {employee_id}:
- Name: {pto_data.get('name', 'Unknown')}
- PTO Balance: {pto_data.get('pto_balance', 'N/A')} days
- PTO Used: {pto_data.get('pto_used', 'N/A')} days
- Total Annual PTO: {pto_data.get('pto_total', 'N/A')} days

Use this data to answer the user's question.
"""
        else:
            pto_context = f"\n\nNote: No PTO data found for employee ID {employee_id}."
    
    system_prompt = f"""You are a PTO (Paid Time Off) specialist assistant. Your role is to answer questions about:
- PTO balances and availability
- Vacation days and time-off requests
- Sick leave policies
- Holiday schedules
- Time-off approval processes
{catalog_context}
{pto_context}

IMPORTANT BOUNDARIES:
- If asked about general company policies (not PTO-related), respond:
  "Please use the Policy bot for general company policy questions."

Provide clear, helpful answers about time off."""
    
    try:
        response = await MODEL_PROVIDER.generate_response(
            system_prompt=system_prompt,
            user_message=question
        )
        return response
    except Exception as e:
        return f"Error from PTO Bot: {str(e)}"


# ============================================================================
# TOOL 3: SERVER INFO
# ============================================================================

@mcp.tool()
async def list_available_tools() -> str:
    """List all available Vendor bot tools and their capabilities."""
    
   # onspring_status = "✓ Connected" if ONSPRING_DATA.enabled else "✗ Not configured (using mock data)"
    
    return f"""
VENDOR BOTS SERVER - Available Tools

1. ASK_POLICY_BOT
   Use for: Company policies, rules, procedures, compliance questions
   Parameters: 
     - question (required): Your policy question
     - include_onspring_data (optional, default=true): Include real policy data
   
2. ASK_PTO_BOT
   Use for: PTO balances, vacation requests, time-off questions
   Parameters:
     - question (required): Your PTO question
     - employee_id (optional): Employee ID for personalized data

3. LIST_AVAILABLE_TOOLS
   Use for: Viewing this help message

Configuration:
- AI Model: {MODEL_PROVIDER.get_provider_name()}
- Required: OPENROUTER_API_KEY
- Optional: ONSPRING_API_KEY, ONSPRING_API_BASE
"""


# ============================================================================
# SERVER RUNNER
# ============================================================================

def main():
    """Start the Vendor Bots MCP server."""
    print("=" * 60)
    print("VENDOR BOTS MCP SERVER")
    print("=" * 60)
    print(f"AI Model: {MODEL_PROVIDER.get_provider_name()}")
    
    print("\n" + "=" * 60)
    print("Server ready!")
    print("=" * 60)
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()