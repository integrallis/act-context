"""DHCM-ng — residual, architecture-grounded, budgeted multi-resolution code context.

See ../DESIGN.md. Clean-slate; no mock models in any evaluated path.
"""

from .model import Kind, Node, Tree, LLMClient, Embedder, content_hash
from .extract import extract_tree
from .llm import OllamaClient, AnthropicClient, cheap_client, frontier_client, load_env
from .summarize import summarize_tree, resummarize_set
from .assemble import assemble_path, assess_locus, redundancy, summary_tokens, path_to

__all__ = [
    "Kind", "Node", "Tree", "LLMClient", "Embedder", "content_hash",
    "extract_tree",
    "OllamaClient", "AnthropicClient", "cheap_client", "frontier_client", "load_env",
    "summarize_tree", "resummarize_set",
    "assemble_path", "assess_locus", "redundancy", "summary_tokens", "path_to",
]
