import asyncio
import os
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# Load environment variables
load_dotenv('.env')

class MCPClient:
    def __init__(self):
        # Initialize OpenAI client
        self.openai = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None


    #server connection function
    #modify to work with kendricks
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        #check file is python, tell to run
        is_python = server_script_path.endswith(".py")
        if not (is_python):
            raise ValueError("Server script must be a .py file")

        command = "python3" if is_python else "error"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        #launch server
        #build write/read streams for communication
        #wrap ^ / handle MCP protocol
        #register them for final cleanup
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        
        #init MCP through handshake w/ server
        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    #Process Query Function
    
    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        #where user input enters (like hw3)
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Get available tools from the MCP server
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in response.tools
        ]

        # Initial request to OpenAI
        #send to open ai ("This is user question, and the tools available for you to use")
        #messages = user question
        #available tools = tools you can use 
        #may need ot update to modern standard for gpt-5
        response = await self.openai.chat.completions.create(
            model="gpt-5-mini",  # You can change this to gpt-4, gpt-3.5-turbo, etc.
            messages=messages,
            tools=available_tools
        )

        # Process tool calls in a loop
        #enter if openai wants to use the tools
        #adds openai response w/ tool calls to convo
        while response.choices[0].message.tool_calls:
            # Add assistant's message to conversation
            messages.append(response.choices[0].message)

            # Execute each tool call
            #AI says: "call tool(get_forcast) with arguments ({"latitude": 123.1, "longitude": -343.1})"
            #extract ^ tool and arguments
            #eval() converts ^(JSON string) to python dictionary
            for tool_call in response.choices[0].message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = eval(tool_call.function.arguments)  # Note: eval is used here for simplicity
                
                print(f"\nCalling tool: {tool_name} with args: {tool_args}")

                # Call the tool via MCP
                #sends message to weather.py server
                result = await self.session.call_tool(tool_name, tool_args)
                #after / bc of ^
                #get_forecast( tool) function runs
                #calls NWS api (weather)
                #gets the weather data
                
                # Add tool result to messages(so openAi can see it)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content)
                })

            # Get next response from OpenAI
            response = await self.openai.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                tools=available_tools
            )

        # Return final response
        #has tool results / data here
        #formulates into natrual language response
        return response.choices[0].message.content

    # Chat Loop function (may replace)
    #
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                response = await self.process_query(query)
                print(f"\n{response}")
                
            except Exception as e:
                print(f"\nError: {str(e)}")

    #final cleanup function
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

#main function 
async def main():
    if len(os.sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        os.sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(os.sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    os.sys = sys
    asyncio.run(main())