import sys, pathlib, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
dummy = types.ModuleType("dummy")
sys.modules.setdefault("pandas", dummy)
sys.modules.setdefault("numpy", dummy)
fastmcp = types.ModuleType("mcp.server.fastmcp")
class FastMCP:
    pass
fastmcp.FastMCP = FastMCP
sys.modules.setdefault("mcp", types.ModuleType("mcp"))
sys.modules.setdefault("mcp.server", types.ModuleType("mcp.server"))
sys.modules.setdefault("mcp.server.fastmcp", fastmcp)
upbit_stub = types.ModuleType("app.tools.upbit")
upbit_stub.get_current_ticker = lambda *a, **k: None
upbit_stub.get_candles_for_daily = lambda *a, **k: None
upbit_stub.get_candles_for_minutes = lambda *a, **k: None
upbit_stub.get_candles_for_weekly = lambda *a, **k: None
sys.modules.setdefault("app.tools.upbit", upbit_stub)

from app.tools.analyze import calculate_distance_to_level

def test_calculate_distance_to_level_relative_to_first_argument():
    assert calculate_distance_to_level(100, 110) == 10
    assert calculate_distance_to_level(100, 90) == -10
    # distance should be relative to the first price
    result = calculate_distance_to_level(120, 100)
    assert result == ((100 - 120) / 120) * 100
