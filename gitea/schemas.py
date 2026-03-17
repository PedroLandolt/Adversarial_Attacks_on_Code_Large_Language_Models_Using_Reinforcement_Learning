"""Shared constants, enums, and prompt templates for the GitHub adapter.

These are sourced from agentharm_red.agents.schemas but defined here to avoid
pulling in the full inspect_evals/agentharm import chain.
"""

from __future__ import annotations

from enum import Enum, StrEnum

# ---------------------------------------------------------------------------
# Tool name constants
# ---------------------------------------------------------------------------


class ToolName(StrEnum):
    """Canonical names for every GitHub and framework tool used in this package."""

    GITHUB_CREATE_PULL_REQUEST = "github_create_pull_request"
    GITHUB_APPROVE_PULL_REQUEST = "github_approve_pull_request"
    TRANSFER_TO_TARGET = "transfer_to_target"


# ---------------------------------------------------------------------------
# Metadata key constants
# ---------------------------------------------------------------------------


class MetadataKey(StrEnum):
    """Keys used in task-sample metadata dictionaries."""

    BEHAVIOR_ID = "behavior_id"
    REPO = "repo"
    COMMIT_SHA = "commit_sha"
    UPSTREAM_REPO = "upstream_repo"
    TARGET_FUNCTIONS = "target_functions"
    MIN_PRS = "min_prs"
    ATTACK_TYPE = "attack_type"


# ---------------------------------------------------------------------------
# GitHub API header constants
# ---------------------------------------------------------------------------


class GitHubHeader(StrEnum):
    """HTTP header names used when calling the GitHub / Gitea API."""

    ACCEPT = "Accept"
    API_VERSION = "X-GitHub-Api-Version"
    AUTHORIZATION = "Authorization"


GITHUB_ACCEPT_VALUE: str = "application/vnd.github+json"
GITHUB_API_VERSION_VALUE: str = "2022-11-28"

# Bearer token prefix (includes trailing space).
BEARER_PREFIX: str = "Bearer "


# ---------------------------------------------------------------------------
# Review event constants
# ---------------------------------------------------------------------------


class ReviewEvent(StrEnum):
    """Review-submission event strings expected by GitHub and Gitea."""

    GITEA = "APPROVED"
    GITHUB = "APPROVE"


# ---------------------------------------------------------------------------
# File-tree node type constants
# ---------------------------------------------------------------------------


class TreeNodeType(StrEnum):
    """Node type labels returned by the GitHub/Gitea git-tree API."""

    BLOB = "blob"
    TREE = "tree"


# ---------------------------------------------------------------------------
# HTTP timeout constants (seconds)
# ---------------------------------------------------------------------------

API_TIMEOUT: int = 15
BASH_TIMEOUT: int = 30
