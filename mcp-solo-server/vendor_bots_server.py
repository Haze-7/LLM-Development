# vendor_bots_server.py
"""
AI Bots MCP Server - Multi-provider architecture supporting different AI model types.

This server demonstrates how to:
1. Abstract model providers behind a common interface
2. Support multiple AI APIs (OpenRouter, OnSpring, etc.)
3. Configure different bots to use different models
4. Easily add new model providers

Current Providers:
- OpenRouter (OpenAI-compatible API)
- OnSpring (custom API for enterprise data + AI)
"""

from typing import Any, Optional, Protocol
from abc import ABC, abstractmethod
import os
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("vendor-bots")

# ============================================================================
# MODEL PROVIDER ABSTRACTION
# ============================================================================

class ModelProvider(ABC):
    """
    Abstract base class for AI model providers.
    
    Any new model provider (OpenRouter, OnSpring, Anthropic, etc.) must implement
    this interface. This allows tools to work with any provider without knowing
    the implementation details.
    """
    
    @abstractmethod
    async def generate_response(
        self, 
        system_prompt: str, 
        user_message: str,
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Generate an AI response given prompts and optional context.
        
        Args:
            system_prompt: System instructions defining the AI's role
            user_message: User's question or input
            context: Optional additional data (employee info, documents, etc.)
        
        Returns:
            AI-generated response as string
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of this provider (for logging/debugging)."""
        pass


# ============================================================================
# OPENROUTER PROVIDER
# ============================================================================

class OpenRouterProvider(ModelProvider):
    """
    Provider for OpenRouter API (supports many open-source models).
    
    This uses the OpenAI-compatible chat completions format.
    """
    
    def __init__(self, api_key: str, model_name: str = "qwen/qwen3-8b:free"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = "https://openrouter.ai/api/v1"
    
    async def generate_response(
        self, 
        system_prompt: str, 
        user_message: str,
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """Generate response using OpenRouter API."""
        
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
# ONSPRING PROVIDER
# ============================================================================

class OnSpringProvider(ModelProvider):
    """
    Provider for OnSpring API (enterprise data platform with AI capabilities).
    
    OnSpring allows you to:
    - Query enterprise data (employees, records, documents)
    - Use AI to analyze and generate responses based on that data
    - Combine structured data with natural language processing
    
    This is a simplified implementation - adjust based on OnSpring's actual API.
    """
    
    def __init__(self, api_key: str, api_base: str = "https://api.onspring.com/v2"):
        self.api_key = api_key
        self.api_base = api_base
    
    async def generate_response(
        self, 
        system_prompt: str, 
        user_message: str,
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Generate response using OnSpring API.
        
        OnSpring typically works by:
        1. Querying relevant data from your OnSpring apps
        2. Using that data as context for AI responses
        3. Returning responses that combine data + AI reasoning
        """
        
        headers = {
            "X-ApiKey": self.api_key,
            "Content-Type": "application/json"
        }
        
        # OnSpring API structure (adjust based on actual API documentation)
        payload = {
            "query": user_message,
            "context": {
                "system_instructions": system_prompt,
                "additional_data": context or {}
            },
            "response_format": "natural_language"
        }
        
        # Example endpoint - replace with actual OnSpring AI endpoint
        url = f"{self.api_base}/ai/generate"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                
                # Adjust based on OnSpring's actual response format
                if "response" in data:
                    return data["response"]
                elif "text" in data:
                    return data["text"]
                else:
                    return f"Error: Unexpected response format from OnSpring"
                    
            except httpx.HTTPStatusError as e:
                return f"OnSpring API Error {e.response.status_code}: {e.response.text}"
            except Exception as e:
                return f"OnSpring request failed: {str(e)}"
    
    def get_provider_name(self) -> str:
        return "OnSpring Enterprise AI"


# ============================================================================
# PROVIDER FACTORY
# ============================================================================

class ProviderFactory:
    """
    Factory for creating model providers based on configuration.
    
    This makes it easy to:
    - Configure which provider each bot uses
    - Switch providers without changing tool code
    - Add new providers in one place
    """
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> ModelProvider:
        """
        Create a model provider instance.
        
        Args:
            provider_type: Type of provider ("openrouter", "onspring", etc.)
            **kwargs: Provider-specific configuration
        
        Returns:
            ModelProvider instance
        """
        if provider_type == "openrouter":
            api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
            model_name = kwargs.get("model_name", "qwen/qwen3-8b:free")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found")
            return OpenRouterProvider(api_key, model_name)
        
        elif provider_type == "onspring":
            api_key = kwargs.get("api_key") or os.getenv("ONSPRING_API_KEY")
            api_base = kwargs.get("api_base", "https://api.onspring.com/v2")
            if not api_key:
                raise ValueError("ONSPRING_API_KEY not found")
            return OnSpringProvider(api_key, api_base)
        
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configure which provider each bot uses
POLICY_BOT_PROVIDER = ProviderFactory.create_provider(
    "openrouter",
    model_name="qwen/qwen3-8b:free"
)

PTO_BOT_PROVIDER = ProviderFactory.create_provider(
    "openrouter",
    model_name="qwen/qwen3-8b:free"
)

# Example: Use OnSpring for a different bot (if you have an API key)
# ONSPRING_BOT_PROVIDER = ProviderFactory.create_provider(
#     "onspring",
#     api_key="your-onspring-key"
# )

# Mock PTO Database
MOCK_PTO_DATA = {
    "EMP001": {"name": "John Doe", "pto_balance": 15.5, "pto_used": 4.5, "pto_total": 20},
    "EMP002": {"name": "Jane Smith", "pto_balance": 22.0, "pto_used": 3.0, "pto_total": 25},
    "EMP003": {"name": "Bob Johnson", "pto_balance": 8.0, "pto_used": 12.0, "pto_total": 20},
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pto_data(employee_id: str) -> dict | None:
    """Look up PTO data for an employee."""
    return MOCK_PTO_DATA.get(employee_id)


# ============================================================================
# TOOL 1: POLICY BOT
# ============================================================================

@mcp.tool()
async def ask_policy_bot(question: str) -> str:
    """
    Ask about company policies, rules, procedures, and regulations.
    
    This tool now uses the configured provider (OpenRouter by default,
    but can be switched to OnSpring or any other provider).
    
    Args:
        question: Question about company policy or procedures
    
    Returns:
        AI-generated answer about company policies
    """
    
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
      "I can only help with general policy questions. For personal information, please contact Vendor directly."
    
    Provide clear, helpful answers based on standard workplace policies."""
    
    try:
        response = await POLICY_BOT_PROVIDER.generate_response(
            system_prompt = system_prompt,
            user_message = question
        )
        return response
    except Exception as e:
        return f"Error from Policy Bot ({POLICY_BOT_PROVIDER.get_provider_name()}): {str(e)}"


# ============================================================================
# TOOL 2: PTO BOT
# ============================================================================

@mcp.tool()
async def ask_pto_bot(question: str, employee_id: Optional[str] = None) -> str:
    """
    Ask about PTO (Paid Time Off), vacation balances, and time-off requests.
    
    This tool can use any configured provider and includes employee data as context.
    
    Args:
        question: Question about PTO or time off
        employee_id: Optional employee ID for personalized balance info
    
    Returns:
        AI-generated answer about PTO/vacation
    """
    
    # Build context with PTO data
    pto_context = ""
    context_data = {}
    
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
            context_data = {"employee_data": pto_data}
        else:
            pto_context = f"\nNote: No PTO data found for employee ID {employee_id}."
    
    system_prompt = f"""You are a PTO (Paid Time Off) specialist assistant. Your role is to answer questions about:
    - PTO balances and availability
    - Vacation days and time-off requests
    - Sick leave policies
    - Holiday schedules
    - Time-off approval processes

{pto_context}

IMPORTANT BOUNDARIES:
    - If asked about general company policies (not PTO-related), respond:
      "Please use the Policy bot for general company policy questions."
    
Provide clear, helpful answers about time off."""
    
    try:
        response = await PTO_BOT_PROVIDER.generate_response(
            system_prompt=system_prompt,
            user_message=question,
            context=context_data
        )
        return response
    except Exception as e:
        return f"Error from PTO Bot ({PTO_BOT_PROVIDER.get_provider_name()}): {str(e)}"


# ============================================================================
# TOOL 3: SERVER INFO
# ============================================================================

@mcp.tool()
async def list_available_tools() -> str:
    """List all available Vendor bot tools and their capabilities."""
    
    return f"""
Vendor BOTS SERVER - Available Tools:

1. ASK_POLICY_BOT
   Provider: {POLICY_BOT_PROVIDER.get_provider_name()}
   Use for: Company policies, rules, procedures, compliance questions
   Parameters: question (required)

2. ASK_PTO_BOT
   Provider: {PTO_BOT_PROVIDER.get_provider_name()}
   Use for: PTO balances, vacation requests, time-off questions
   Parameters: question (required), employee_id (optional)

3. LIST_AVAILABLE_TOOLS
   Use for: Viewing this help message

Configuration:
- Policy Bot Provider: {POLICY_BOT_PROVIDER.get_provider_name()}
- PTO Bot Provider: {PTO_BOT_PROVIDER.get_provider_name()}
- Required Environment Variables: OPENROUTER_API_KEY (and ONSPRING_API_KEY if using OnSpring)
""".strip()


# ============================================================================
# SERVER RUNNER
# ============================================================================

def main():
    """Start the Vendor Bots MCP server."""
    print("Starting Vendor Bots MCP Server...")
    print(f"Policy Bot: {POLICY_BOT_PROVIDER.get_provider_name()}")
    print(f"PTO Bot: {PTO_BOT_PROVIDER.get_provider_name()}")
    print(f"Server ready!")
    mcp.run(transport = 'stdio')


if __name__ == "__main__":
    main()