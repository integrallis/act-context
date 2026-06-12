"""DHCM-ng core model: the structural tree + protocols (the shared contract).

Everything else (extract, summarize, index, descent, eval) builds on this. Kept
deliberately small and dependency-free so the contract is stable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class Kind(str, Enum):
    SUBSYSTEM = "subsystem"   # repo root
    COMPONENT = "component"   # top-level package / directory
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"     # function or method (leaf)


@dataclass
class Node:
    id: str
    kind: Kind
    name: str
    path: str                      # file path; dir for component/subsystem
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    source: str = ""               # source code for class/function leaves
    signature: str = ""
    docstring: str = ""
    start_line: int = 0
    end_line: int = 0
    content_hash: str = ""         # hash of own source (leaf) — for incremental rebuild
    subtree_hash: str = ""         # hash over subtree (own + children hashes)
    # Summaries (filled by summarizers). Two schemes compared in the H2' ablation:
    residual_summary: str = ""     # local delta only; ancestors carry global context
    cumulative_summary: str = ""   # self-contained (restates child/descendant context)


@dataclass
class Tree:
    nodes: dict[str, Node]
    root: str

    def get(self, nid: str) -> Node:
        return self.nodes[nid]

    def children(self, nid: str) -> list[Node]:
        return [self.nodes[c] for c in self.nodes[nid].children]

    def ancestors(self, nid: str) -> list[Node]:
        """Nearest-first list of ancestor nodes (excludes self)."""
        out: list[Node] = []
        cur = self.nodes[nid].parent
        while cur is not None:
            out.append(self.nodes[cur])
            cur = self.nodes[cur].parent
        return out

    def descendants(self, nid: str) -> list[Node]:
        out: list[Node] = []
        stack = list(self.nodes[nid].children)
        while stack:
            c = stack.pop()
            out.append(self.nodes[c])
            stack.extend(self.nodes[c].children)
        return out

    def leaves_under(self, nid: str) -> list[Node]:
        return [n for n in self.descendants(nid) if not n.children] or (
            [self.nodes[nid]] if not self.nodes[nid].children else []
        )

    def by_kind(self, kind: Kind) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == kind]

    def internal_nodes(self) -> list[Node]:
        """Nodes that need a generated summary (have children)."""
        return [n for n in self.nodes.values() if n.children]


class LLMClient(Protocol):
    """Real, cheap model (e.g. local Ollama). NO mock in any evaluated path."""

    def generate(self, prompt: str, max_tokens: int = 256) -> str: ...


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]
