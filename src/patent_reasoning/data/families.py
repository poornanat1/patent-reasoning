"""Application families, so that related applications never land in different splits.

Two applications are in the same family when they're linked, directly or through other
applications, by:

- continuity: continuation, divisional, continuation-in-part, or national stage entry of
  the same PCT application (the ODP ``continuity`` endpoint, parents and children)
- a shared foreign priority filing (the ODP ``foreign-priority`` endpoint)

This is broader than a DOCDB family, which errs on the side of keeping related claims
together.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from patent_reasoning.data.odp import ODPClient


def _bag(data: dict[str, Any] | None, key: str) -> list[dict[str, Any]]:
    bags = (data or {}).get("patentFileWrapperDataBag") or [{}]
    return bags[0].get(key) or []


def family_links(client: ODPClient, app_id: str) -> set[str]:
    """Nodes an application is linked to: related applications and priority filings."""
    links = {f"app:{app_id}"}
    continuity = client.get_json(f"patent/applications/{app_id}/continuity")
    for item in _bag(continuity, "parentContinuityBag"):
        if parent := item.get("parentApplicationNumberText"):
            links.add(f"app:{parent}")
    for item in _bag(continuity, "childContinuityBag"):
        if child := item.get("childApplicationNumberText"):
            links.add(f"app:{child}")
    priority = client.get_json(f"patent/applications/{app_id}/foreign-priority")
    for item in _bag(priority, "foreignPriorityBag"):
        number = re.sub(r"[^0-9A-Za-z]", "", item.get("applicationNumberText") or "")
        if number:
            links.add(f"priority:{item.get('ipOfficeName', '')}:{number}")
    return links


def families(links: dict[str, set[str]]) -> dict[str, str]:
    """Application -> family ID, from each application's link set (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for nodes in links.values():
        nodes = sorted(nodes)
        for other in nodes[1:]:
            a, b = find(nodes[0]), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
    members: dict[str, list[str]] = {}
    for app in links:
        members.setdefault(find(f"app:{app}"), []).append(app)
    out = {}
    for apps in members.values():
        family = "fam-" + hashlib.sha256(",".join(sorted(apps)).encode()).hexdigest()[:12]
        out.update(dict.fromkeys(apps, family))
    return out
