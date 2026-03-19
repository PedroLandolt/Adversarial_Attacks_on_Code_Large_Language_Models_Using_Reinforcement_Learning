#!/usr/bin/env bash
# Set up a local Gitea instance
#
# Usage:
#   ./scripts/setup_gitea.sh [repo-name]
#
# What this does:
#   1. Starts Gitea via docker compose (if not already running)
#   2. Creates an admin user (admin / adminpass123)
#   3. Creates a personal access token
#   4. Creates the target repo with an initial README on main
#
# Note: no PR is seeded — the executor agent creates PRs during the benchmark run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/../gitea/docker-compose.yml"
GITEA_URL="http://localhost:3001"
ADMIN_USER="gitadmin"
ADMIN_PASS="adminpass123"
ADMIN_EMAIL="gitadmin@local.dev"
REVIEWER_USER="reviewer"
REVIEWER_PASS="reviewerpass123"
REVIEWER_EMAIL="reviewer@local.dev"
REPO_NAME="${1:-test-repo}"
TOKEN_NAME="decomp-local"
REVIEWER_TOKEN_NAME="reviewer-local"

# ---------------------------------------------------------------------------
# 1. Start Gitea
# ---------------------------------------------------------------------------
echo "[1/4] Starting Gitea..."
docker compose -f "$COMPOSE_FILE" up -d

echo "      Waiting for Gitea to be ready..."
for i in $(seq 1 30); do
    if curl -sf "${GITEA_URL}/api/healthz" > /dev/null 2>&1; then
        echo "      Gitea is up."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Gitea did not become healthy in time." >&2
        exit 1
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# 2. Create admin user (idempotent)
# ---------------------------------------------------------------------------
echo "[2/4] Creating admin user '${ADMIN_USER}'..."
docker compose -f "$COMPOSE_FILE" exec -T --user git gitea \
    gitea admin user create \
    --username "$ADMIN_USER" \
    --password "$ADMIN_PASS" \
    --email "$ADMIN_EMAIL" \
    --admin \
    --must-change-password=false 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Create access token (delete old one with same name first)
# ---------------------------------------------------------------------------
echo "[3/4] Creating access token..."

existing_id=$(curl -sf -u "${ADMIN_USER}:${ADMIN_PASS}" \
    "${GITEA_URL}/api/v1/users/${ADMIN_USER}/tokens" \
    | python3 -c "
import sys, json
toks = [t['id'] for t in json.load(sys.stdin) if t['name'] == '${TOKEN_NAME}']
print(toks[0] if toks else '')
" 2>/dev/null || true)

if [ -n "$existing_id" ]; then
    curl -sf -X DELETE -u "${ADMIN_USER}:${ADMIN_PASS}" \
        "${GITEA_URL}/api/v1/users/${ADMIN_USER}/tokens/${existing_id}" > /dev/null
fi

TOKEN=$(curl -sf -X POST \
    -H "Content-Type: application/json" \
    -u "${ADMIN_USER}:${ADMIN_PASS}" \
    "${GITEA_URL}/api/v1/users/${ADMIN_USER}/tokens" \
    -d "{\"name\":\"${TOKEN_NAME}\",\"scopes\":[\"write:repository\",\"write:issue\",\"write:user\"]}" \
    | python3 -c "import sys, json; print(json.load(sys.stdin)['sha1'])")

if [ -z "$TOKEN" ]; then
    echo "ERROR: Failed to create access token." >&2
    exit 1
fi
echo "      Token created."

# ---------------------------------------------------------------------------
# 4. Create reviewer user + token (separate account so it can approve PRs)
# ---------------------------------------------------------------------------
echo "[4/5] Creating reviewer user '${REVIEWER_USER}'..."
docker compose -f "$COMPOSE_FILE" exec -T --user git gitea \
    gitea admin user create \
    --username "$REVIEWER_USER" \
    --password "$REVIEWER_PASS" \
    --email "$REVIEWER_EMAIL" \
    --must-change-password=false 2>/dev/null || true

existing_reviewer_id=$(curl -sf -u "${REVIEWER_USER}:${REVIEWER_PASS}" \
    "${GITEA_URL}/api/v1/users/${REVIEWER_USER}/tokens" \
    | python3 -c "
import sys, json
toks = [t['id'] for t in json.load(sys.stdin) if t['name'] == '${REVIEWER_TOKEN_NAME}']
print(toks[0] if toks else '')
" 2>/dev/null || true)

if [ -n "$existing_reviewer_id" ]; then
    curl -sf -X DELETE -u "${REVIEWER_USER}:${REVIEWER_PASS}" \
        "${GITEA_URL}/api/v1/users/${REVIEWER_USER}/tokens/${existing_reviewer_id}" > /dev/null
fi

REVIEWER_TOKEN=$(curl -sf -X POST \
    -H "Content-Type: application/json" \
    -u "${REVIEWER_USER}:${REVIEWER_PASS}" \
    "${GITEA_URL}/api/v1/users/${REVIEWER_USER}/tokens" \
    -d "{\"name\":\"${REVIEWER_TOKEN_NAME}\",\"scopes\":[\"write:repository\",\"write:issue\"]}" \
    | python3 -c "import sys, json; print(json.load(sys.stdin)['sha1'])")

if [ -z "$REVIEWER_TOKEN" ]; then
    echo "ERROR: Failed to create reviewer token." >&2
    exit 1
fi
echo "      Reviewer token created."

# ---------------------------------------------------------------------------
# 5. Create repo (auto-initialised with README on main)
# ---------------------------------------------------------------------------
echo "[5/5] Creating repository '${ADMIN_USER}/${REPO_NAME}'..."
http_status=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST \
    -H "Authorization: token ${TOKEN}" \
    -H "Content-Type: application/json" \
    "${GITEA_URL}/api/v1/user/repos" \
    -d "{\"name\":\"${REPO_NAME}\",\"auto_init\":true,\"default_branch\":\"main\",\"private\":false}")

if [ "$http_status" = "409" ]; then
    echo "      Repository already exists — skipping."
elif [ "$http_status" = "201" ]; then
    echo "      Repository created."
else
    echo "ERROR: Unexpected status ${http_status} when creating repo." >&2
    exit 1
fi

# Add reviewer as collaborator with write access (needed to post reviews)
curl -sf -X PUT \
    -H "Authorization: token ${TOKEN}" \
    -H "Content-Type: application/json" \
    "${GITEA_URL}/api/v1/repos/${ADMIN_USER}/${REPO_NAME}/collaborators/${REVIEWER_USER}" \
    -d '{"permission":"write"}' > /dev/null && echo "      Reviewer added as collaborator."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<EOF

==========================================
  Local Gitea ready
==========================================

  Repo:     ${GITEA_URL}/${ADMIN_USER}/${REPO_NAME}
  Executor: ${ADMIN_USER} / ${ADMIN_PASS}
  Reviewer: ${REVIEWER_USER} / ${REVIEWER_PASS}

  Export these before running the benchmark:

    export GITHUB_TOKEN=${TOKEN}
    export REVIEWER_TOKEN=${REVIEWER_TOKEN}
    export GITHUB_API_URL=${GITEA_URL}/api/v1
    export GITEA_REPO=${ADMIN_USER}/${REPO_NAME}

EOF
