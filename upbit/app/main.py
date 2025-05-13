import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

from app.tools.upbit import set_tools as set_upbit_tools
from app.tools.analyze import set_tools as set_analyze_tools

# load .env file
load_dotenv()

mcp = FastMCP("Cashy", stateless_http=True)
set_upbit_tools(mcp)
set_analyze_tools(mcp)

if __name__ == "__main__":
    mode = os.getenv("MODE", "stdio")

    if mode == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")