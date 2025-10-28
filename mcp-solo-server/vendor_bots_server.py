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
# ONSPRING APP CATALOG - YOU MAINTAIN THIS
# ============================================================================

class OnSpringAppCatalog:
    """
    Manual catalog of OnSpring apps with REAL field name examples.
    
    ✏️ REPLACE temp values with your actual OnSpring data:
    - App IDs from OnSpring
    - Real field names from each app
    - Accurate descriptions
    """
    
    APPS = {      
        # ================================================================
        # POLICIES TABLE
        # ================================================================
        "policies": {
            "id": 74,
            "description": "Company policies, procedures, handbooks, and compliance documents",
            "example_fields": {
                "Policy Category": 5141,  # Reference type - cannot filter
                "Purpose": 4785,          # Text field - can filter with eq, ne
                "Policy Owner": 4772      # Reference type - cannot filter
            },
            "filter_field_id": 4785,  # Purpose field (text type)
            "filter_field_type": "text"
        },
        
        # ================================================================
        # Policy Categories TABLE
        # ================================================================
        "policy_categories": {
            "id": 81,
            "description": "Categories for organizing company policies and documents",
            "example_fields": {
                "record ID": 5123,
                "Category": 5130
            },
            "filter_field_id": 5123,
            "filter_field_type": "numeric"
        },
        
        # ================================================================
        # Policy Subcategories TABLE
        # ================================================================
        "policy_subcategories": {
            "id": 82,
            "description": "Subcategories for organizing company policies and documents",
            "example_fields": {
                "Record ID": 5131,
                "Subcategory": 5183,   
                "Category_Ref": 5139,
            },
            "filter_field_id": 5183,
            "filter_field_type": "text"
        },
        
        # ================================================================
        # Facilities TABLE
        # ================================================================
        "facilities": {
            "id": 75,
            "description": "Facilities and locations within the organization",
            "example_fields": {
                "Display Name": 5109,
                "Record Id": 4847,   
            },
            "filter_field_id": 5109,
            "filter_field_type": "text"
        },
        
        # ================================================================
        # Users TABLE
        # ================================================================
        "users": {
            "id": 2,
            "description": "Users and personnel within the organization",
            "example_fields": {
                "Full Name": 46,
                "Email Address": 33,   
                "Record Id": 5
            },
            "filter_field_id": 46,
            "filter_field_type": "text"
        },
    }
    
    @classmethod
    def get_app_info(cls, app_name: str) -> Optional[dict]:
        """Get info about a specific app/table."""
        return cls.APPS.get(app_name)
    
    @classmethod
    def list_all_apps(cls) -> dict[str, dict]:
        """Get all available apps/tables."""
        return cls.APPS
    
    @classmethod
    def format_catalog_for_prompt(cls) -> str:
        """Format the catalog into text for AI system prompts."""
        lines = ["\nAVAILABLE DATA SOURCES IN ONSPRING:"]
        
        for app_name, app_info in cls.APPS.items():
            lines.append(f"\n{app_name.upper()}:")
            lines.append(f"  Purpose: {app_info['description']}")
            
            if isinstance(app_info['example_fields'], dict):
                example_fields_str = ", ".join(app_info['example_fields'].keys())
            else:
                example_fields_str = ", ".join(app_info['example_fields'])
            
            lines.append(f"  Example fields: {example_fields_str}")
            lines.append("  [Full field list will be fetched when querying this table]")
        
        return "\n".join(lines)


# ============================================================================
# ONSPRING SCHEMA MANAGER - DYNAMIC FIELD DISCOVERY
# ============================================================================

class OnSpringSchemaManager:
    """
    Fetches complete field schemas from OnSpring API dynamically.
    """
    
    def __init__(
        self, 
        api_key: str, 
        api_base: str = "https://api.onspring.com",
        cache_duration_minutes: int = 60
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        
        self._schema_cache: dict[str, dict] = {}
        self._cache_timestamps: dict[str, datetime] = {}
        
        # Add sensitive fields to exclude if needed
        self.excluded_fields: dict[str, list[str]] = {
            # Example: "employees": ["ssn", "salary", "bank_account"],
        }
    
    async def get_full_schema(
        self, 
        app_name: str, 
        force_refresh: bool = False
    ) -> Optional[dict[str, Any]]:
        """Fetch ALL fields for a specific table."""
        
        # Check cache
        if not force_refresh and self._is_cached(app_name):
            print(f"[Schema] Using cached schema for {app_name}")
            return self._schema_cache[app_name]
        
        # Get app info from catalog
        app_info = OnSpringAppCatalog.get_app_info(app_name)
        if not app_info:
            print(f"[Schema] Unknown app: {app_name}")
            return None
        
        # Fetch from API
        print(f"[Schema] Fetching schema for {app_name} from OnSpring...")
        schema = await self._fetch_schema_from_api(app_info["id"])
        
        if schema:
            schema = self._filter_excluded_fields(app_name, schema)
            self._schema_cache[app_name] = schema
            self._cache_timestamps[app_name] = datetime.now()
            print(f"[Schema] Cached {len(schema['fields'])} fields for {app_name}")
        
        return schema
    
    async def _fetch_schema_from_api(self, app_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch field list from OnSpring API.
        
        Endpoint: GET /Apps/{appId}/Fields
        
        Note: Some OnSpring instances require specific permissions.
        If this fails with 404, the API key may not have field-level access.
        """
        headers = {
            "X-ApiKey": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Ensure app_id is an integer
        app_id = int(app_id) if isinstance(app_id, str) else app_id
        
        # Try the standard Fields endpoint first
        url = f"{self.api_base}/Apps/{app_id}/Fields"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                return self._transform_schema(data)
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    print(f"[Schema] App {app_id} fields not accessible (404)")
                    print(f"[Schema] This may be a permission issue with the API key")
                    print(f"[Schema] Continuing without schema - queries will still work")
                    # Return empty schema instead of None to allow queries to continue
                    return {"fields": {}}
                else:
                    print(f"[Schema] HTTP Error {e.response.status_code}: {e.response.text}")
                    return {"fields": {}}
            except Exception as e:
                print(f"[Schema] Error fetching schema: {e}")
                return {"fields": {}}
    
    def _transform_schema(self, api_response: dict) -> dict[str, Any]:
        """
        Transform OnSpring response to standard format.
        
        OnSpring v2 returns paginated results with 'items' array containing fields.
        """
        schema = {"fields": {}}
        
        # Handle paginated response
        items = api_response.get("items", [])
        
        for field in items:
            field_name = field.get("name")
            
            if field_name:
                schema["fields"][field_name] = {
                    "id": field.get("id"),
                    "type": field.get("type", "unknown").lower(),
                    "status": field.get("status", "Unknown"),
                    "required": field.get("isRequired", False),
                    "unique": field.get("isUnique", False),
                }
        
        return schema
    
    def _filter_excluded_fields(self, app_name: str, schema: dict) -> dict:
        """Remove sensitive fields."""
        if app_name in self.excluded_fields:
            for excluded_field in self.excluded_fields[app_name]:
                schema["fields"].pop(excluded_field, None)
        return schema
    
    def _is_cached(self, app_name: str) -> bool:
        """Check if schema is cached and valid."""
        if app_name not in self._cache_timestamps:
            return False
        age = datetime.now() - self._cache_timestamps[app_name]
        return age < self.cache_duration
    
    def format_schema_for_prompt(self, app_name: str, schema: dict) -> str:
        """Format schema for AI prompts."""
        if not schema or "fields" not in schema:
            return ""
        
        lines = [f"\nCOMPLETE FIELDS FOR {app_name.upper()}:"]
        
        for field_name, field_info in schema["fields"].items():
            field_type = field_info.get("type", "unknown")
            required = " (required)" if field_info.get("required") else ""
            status = field_info.get("status", "")
            
            line = f"  - {field_name} (ID: {field_info.get('id')}, Type: {field_type}){required}"
            if status and status != "Enabled":
                line += f" [{status}]"
            lines.append(line)
        
        return "\n".join(lines)


# ============================================================================
# ONSPRING DATA SOURCE
# ============================================================================

class OnSpringDataSource:
    """Main interface for querying OnSpring data."""
    
    def __init__(self, api_key: str, api_base: str = "https://api.onspring.com"):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.enabled = bool(api_key)
        self.schema = OnSpringSchemaManager(api_key, api_base) if self.enabled else None
    
    def get_catalog_context(self) -> str:
        """Get catalog formatted for AI prompts."""
        return OnSpringAppCatalog.format_catalog_for_prompt()
    
    async def query_app(
        self,
        app_name: str,
        record_id: Optional[str] = None,
        filter_expression: Optional[str] = None,
        field_ids: Optional[list[int]] = None,
        data_format: str = "Formatted"
    ) -> Optional[dict[str, Any]]:
        """
        Query an OnSpring app/table.
        
        Args:
            app_name: Name of app from catalog
            record_id: Specific record ID to fetch (optional)
            filter_expression: OnSpring filter syntax (optional)
            field_ids: Specific field IDs to return (optional, returns all if None)
            data_format: "Raw" or "Formatted" (default: Formatted)
        
        Returns:
            Dictionary with query results and metadata
        """
        if not self.enabled:
            return None
        
        app_info = OnSpringAppCatalog.get_app_info(app_name)
        if not app_info:
            print(f"[Data] Unknown app: {app_name}")
            return None
        
        # Fetch schema (cached after first call)
        schema = await self.schema.get_full_schema(app_name)
        
        headers = {
            "X-ApiKey": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Route to appropriate endpoint
        app_id = int(app_info['id']) if isinstance(app_info['id'], str) else app_info['id']
        
        if record_id:
            # GET single record
            url = f"{self.api_base}/Records/{record_id}"
            params = {"appId": app_id}  # Now guaranteed to be int
            if field_ids:
                params["fieldIds"] = ",".join(map(str, field_ids))
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(url, headers=headers, params=params, timeout=30.0)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Wrap single record in consistent format
                    result = {
                        "items": [data],
                        "totalRecords": 1,
                        "_schema": schema,
                        "_app_name": app_name
                    }
                    return result
                    
                except httpx.HTTPStatusError as e:
                    print(f"[Data] HTTP Error {e.response.status_code}: {e.response.text}")
                    return None
                except Exception as e:
                    print(f"[Data] Error fetching record: {e}")
                    return None
        else:
            # POST query for multiple records
            url = f"{self.api_base}/Records/Query"
            payload = {
                "appId": app_id,
                "dataFormat": data_format
            }
            
            # OnSpring v2 requires a filter field, even if empty
            # Use a filter that matches all records if no filter provided
            if filter_expression:
                payload["filter"] = filter_expression
            else:
                # Use the designated filter_field_id from catalog (non-reference field)
                filter_field_id = app_info.get('filter_field_id')
                filter_field_type = app_info.get('filter_field_type', 'text')  # Default to text
                
                if filter_field_id:
                    # Choose filter based on field type
                    if filter_field_type == 'numeric':
                        # For numeric fields, use "gt 0" or "ge 0"
                        payload["filter"] = f"{filter_field_id} gt 0"
                    else:
                        # For text fields, use "ne ''" (not equal to empty string)
                        # This matches all records with any value in that field
                        payload["filter"] = f"{filter_field_id} ne ''"
                    
                    print(f"[Data] Using default filter: {payload['filter']}")
                else:
                    # Fallback: try to find a non-reference field
                    example_fields = app_info.get('example_fields', {})
                    if example_fields:
                        # Use the second field (first might be reference)
                        field_ids_list = list(example_fields.values())
                        # Try field at index 1 if exists, otherwise use first
                        fallback_id = field_ids_list[1] if len(field_ids_list) > 1 else field_ids_list[0]
                        payload["filter"] = f"{fallback_id} ne ''"
                        print(f"[Data] Warning: No filter_field_id specified for {app_name}, using {fallback_id} ne ''")
            
            if field_ids:
                payload["fieldIds"] = field_ids
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Attach metadata
                    data["_schema"] = schema
                    data["_app_name"] = app_name
                    
                    return data
                    
                except httpx.HTTPStatusError as e:
                    print(f"[Data] HTTP Error {e.response.status_code}: {e.response.text}")
                    return None
                except Exception as e:
                    print(f"[Data] Error querying {app_name}: {e}")
                    return None
    
    async def get_employee_data(self, employee_id: str) -> Optional[dict[str, Any]]:
        """Query employee/user table by record ID."""
        return await self.query_app("users", record_id=employee_id)
    
    async def get_policy_data(
        self, 
        filter_expression: Optional[str] = None,
        limit: int = 10
    ) -> Optional[dict[str, Any]]:
        """
        Query policies table.
        
        Args:
            filter_expression: OnSpring filter syntax (e.g., "4785 eq 'safety'")
            limit: Number of records to return (for display purposes)
        """
        return await self.query_app("policies", filter_expression=filter_expression)


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

# Initialize OnSpring Data Source
ONSPRING_DATA = OnSpringDataSource(
    api_key=os.getenv("ONSPRING_API_KEY", ""),
    api_base=os.getenv("ONSPRING_API_BASE", "https://api.onspring.com")
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
    """Get PTO data - tries OnSpring first, falls back to mock."""
    if ONSPRING_DATA.enabled:
        employee_data = await ONSPRING_DATA.get_employee_data(employee_id)
        if employee_data:
            return employee_data
    
    return MOCK_PTO_DATA.get(employee_id)


async def get_policy_context() -> str:
    """Get policy context from OnSpring for AI prompts."""
    if not ONSPRING_DATA.enabled:
        return ""
    
    policy_data = await ONSPRING_DATA.get_policy_data()
    if not policy_data:
        return ""
    
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
    
    if include_onspring_data and ONSPRING_DATA.enabled:
        catalog_context = ONSPRING_DATA.get_catalog_context()
        policy_context = await get_policy_context()
    
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
    if ONSPRING_DATA.enabled:
        catalog_context = ONSPRING_DATA.get_catalog_context()
    
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
    
    onspring_status = "✓ Connected" if ONSPRING_DATA.enabled else "✗ Not configured (using mock data)"
    
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
- OnSpring Data: {onspring_status}
- OnSpring API Base: {ONSPRING_DATA.api_base}
- Required: OPENROUTER_API_KEY
- Optional: ONSPRING_API_KEY, ONSPRING_API_BASE

OnSpring Tables Configured:
{chr(10).join(f'  - {name}: {info["description"]}' for name, info in OnSpringAppCatalog.list_all_apps().items())}
""".strip()


# ============================================================================
# SERVER RUNNER
# ============================================================================

def main():
    """Start the Vendor Bots MCP server."""
    print("=" * 60)
    print("VENDOR BOTS MCP SERVER")
    print("=" * 60)
    print(f"AI Model: {MODEL_PROVIDER.get_provider_name()}")
    print(f"OnSpring: {'✓ Connected' if ONSPRING_DATA.enabled else '✗ Not configured (using mock data)'}")
    
    if ONSPRING_DATA.enabled:
        print(f"OnSpring API: {ONSPRING_DATA.api_base}")
        print(f"\nOnSpring Tables:")
        for name, info in OnSpringAppCatalog.list_all_apps().items():
            if isinstance(info['example_fields'], dict):
                field_count = len(info['example_fields'])
            else:
                field_count = len(info['example_fields'])
            print(f"  - {name}: {field_count} example fields")
    
    print("\n" + "=" * 60)
    print("Server ready!")
    print("=" * 60)
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()