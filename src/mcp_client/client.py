import requests
from typing import Any, Dict, List

class MCPClient:
    """Python MCP-Client (HTTP)."""
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.base = server_url.rstrip("/")

    def discover(self) -> List[str]:
        r = requests.get(f"{self.base}/mcp/discover")
        r.raise_for_status()
        return r.json()["tools"]

    def invoke(self, tool: str, args: Dict[str, Any]) -> Any:
        payload = {"tool": tool, "args": args}
        r = requests.post(f"{self.base}/mcp/invoke", json=payload)
        r.raise_for_status()
        return r.json()["result"]
