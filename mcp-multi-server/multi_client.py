import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# Load environment variables
load_dotenv('.env')

class MultiServerMCPClient:
    def __init__(self):
        # Initialize OpenAI client
        self.openai = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.exit_stack = AsyncExitStack()
        self.sessions = {}  # Store multiple server sessions: {name: session}
        self.server_info = {}  # Store server metadata: {name: {tools, transport}}

    async def connect_to_server(self, name: str, server_script_path: str):
        """Connect to an MCP server and register it with a name.
        
        Args:
            name: Unique identifier for this server (e.g., "policy-bot", "pto-bot")
            server_script_path: Path to the server script (.py file)
        """
        # Validate file type
        if not server_script_path.endswith(".py"):
            raise ValueError("Server script must be a .py file")

        print(f"\n🔌 Connecting to {name}...")

        # Set up server parameters
        server_params = StdioServerParameters(
            command="python3",
            args=[server_script_path],
            env=None
        )

        # Launch server and create communication streams
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        
        # Create MCP session
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        
        # Initialize connection with server
        await session.initialize()

        # Store session
        self.sessions[name] = session
        
        # Get and store available tools
        response = await session.list_tools()
        tools = response.tools
        self.server_info[name] = {
            "tools": tools,
            "stdio": stdio,
            "write": write
        }
        
        tool_names = [tool.name for tool in tools]
        print(f"✅ {name} connected with tools: {tool_names}")

    async def get_all_tools(self) -> list[dict]:
        """Gather all tools from all connected servers for OpenAI."""
        all_tools = []
        
        for server_name, info in self.server_info.items():
            for tool in info["tools"]:
                all_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
        
        return all_tools

    async def call_tool(self, tool_name: str, tool_args: dict):
        """Find which server has the requested tool and call it.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        # Search through all servers to find the tool
        for server_name, info in self.server_info.items():
            tool_names = [t.name for t in info["tools"]]
            if tool_name in tool_names:
                print(f"  📡 Routing to {server_name}")
                session = self.sessions[server_name]
                return await session.call_tool(tool_name, tool_args)
        
        # Tool not found in any server
        raise ValueError(f"Tool '{tool_name}' not found in any connected server")

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI with all available tools from all servers.
        
        Args:
            query: User's question or request
            
        Returns:
            Final response from OpenAI
        """
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Get all available tools from all connected servers
        available_tools = await self.get_all_tools()

        # Initial request to OpenAI with all available tools
        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini (updated from gpt-5-mini)
            messages=messages,
            tools=available_tools
        )

        # Process tool calls in a loop (OpenAI might need multiple tool calls)
        while response.choices[0].message.tool_calls:
            # Add assistant's response to conversation
            messages.append(response.choices[0].message)

            # Execute each tool call
            for tool_call in response.choices[0].message.tool_calls:
                tool_name = tool_call.function.name
                
                # Safely parse arguments
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                
                print(f"\n🔧 Calling tool: {tool_name}")
                print(f"   Arguments: {tool_args}")

                # Call the tool via appropriate MCP server
                try:
                    result = await self.call_tool(tool_name, tool_args)
                    result_content = str(result.content)
                except Exception as e:
                    result_content = f"Error calling tool: {str(e)}"
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content
                })

            # Get next response from OpenAI
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=available_tools
            )

        # Return final response
        return response.choices[0].message.content

    async def chat_loop(self):
        """Run an interactive chat loop with access to all connected servers."""
        print("\n" + "="*60)
        print("🤖 Multi-Server MCP Client Started!")
        print("="*60)
        print(f"\nConnected servers: {list(self.sessions.keys())}")
        print("\nType your queries or 'quit' to exit.")
        print("="*60)
        
        while True:
            try:
                query = input("\n💬 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n🤔 Processing...")
                response = await self.process_query(query)
                print(f"\n✨ Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    async def cleanup(self):
        """Clean up all server connections and resources."""
        print("\n🧹 Cleaning up connections...")
        await self.exit_stack.aclose()

async def main():
    """Main entry point for the multi-server client."""
    if len(os.sys.argv) < 2:
        print("Usage: python multi_client.py <server1.py> <server2.py> ...")
        print("\nExample:")
        print("  python multi_client.py policy_bot_server.py pto_bot_server.py")
        os.sys.exit(1)

    client = MultiServerMCPClient()
    
    try:
        # Connect to all servers provided as command line arguments
        server_paths = os.sys.argv[1:]
        
        for i, server_path in enumerate(server_paths):
            # Extract a name from the file path (e.g., "policy_bot_server.py" -> "policy-bot")
            server_name = os.path.basename(server_path).replace("_server.py", "").replace("_", "-")
            await client.connect_to_server(server_name, server_path)
        
        # Start interactive chat loop
        await client.chat_loop()
        
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    os.sys = sys
    asyncio.run(main())

#run Commands setup:
"""
python multi_client.py policy_bot_server.py pto_bot_server.py

Laptop: python3 multi_client.py policy_bot_server.py pto_bot_server.py
```

### Step 3: Test it out!

**Example conversations:**
```
💬 Query: What's the company's remote work policy?
# → GPT routes to ask_policy_bot
# → Policy Bot (Qwen) explains remote work policies

💬 Query: How many vacation days do I have left?
# → GPT routes to ask_pto_bot
# → PTO Bot responds (general or asks for employee ID)

💬 Query: Check PTO for employee EMP001
# → GPT routes to ask_pto_bot with employee_id="EMP001"
# → PTO Bot returns: "John Doe has 15.5 days remaining"
```

## 🎯 Key Features:

### Intelligent Routing
GPT-4o-mini sees both tools and automatically picks the right one based on:
- Tool descriptions
- Question context
- Past conversation

### Boundary Enforcement
Each bot has system prompts that refuse out-of-scope questions:
- Policy Bot: "Use PTO bot for time-off questions"
- PTO Bot: "Use Policy bot for general policies"

This forces proper routing and prevents overlap!

### Mock Data (PTO Bot)
The PTO bot has 3 mock employees:
- **EMP001**: John Doe - 15.5 days left
- **EMP002**: Jane Smith - 22.0 days left
- **EMP003**: Bob Johnson - 8.0 days left

You can easily replace this with a real database!

## 🔍 What Happens Behind the Scenes:
```
User: "What's our vacation policy?"
    ↓
GPT-4o-mini: "This is about policy, I should use ask_policy_bot"
    ↓
Client: Routes to policy_bot_server.py
    ↓
Qwen3-8B: Generates policy explanation
    ↓
Client: Returns result to GPT
    ↓
GPT-4o-mini: Formats natural response
    ↓
User: Sees final answer
"""