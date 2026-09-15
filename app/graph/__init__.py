"""
The agent graph: classify, extract, retrieve, plan.

The graph proposes only. Nothing under this package can reach storage or run a
tool; it reads a message and returns a checked proposal, and the caller decides
what to record.
"""
