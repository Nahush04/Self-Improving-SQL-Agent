"""Memory MCP server: a standalone store of lessons, example queries, and episodes.

The agent talks to this over MCP, the same way any other MCP client would. Nothing here
imports from the agent package — the two only ever communicate through MCP tool calls.
"""
