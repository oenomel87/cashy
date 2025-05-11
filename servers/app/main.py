import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

from app.tools.upbit import get_current_ticker, get_candles_for_daily, get_candles_for_minutes, get_candles_for_weekly

# load .env file
load_dotenv()

mcp = FastMCP("Cashy", stateless_http=True)
mcp.add_tool(get_current_ticker, "get_current_ticker")
mcp.add_tool(get_candles_for_daily, "get_candles_for_daily")
mcp.add_tool(get_candles_for_weekly, "get_candles_for_weekly")
mcp.add_tool(get_candles_for_minutes, "get_candles_for_minutes")

if __name__ == "__main__":
    mode = os.getenv("MODE", "stdio")

    if mode == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")