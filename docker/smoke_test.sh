#!/usr/bin/env bash
# =============================================================================
# Smoke Test: NVIDIA Metropolis AI Surveillance Docker Stack
# Builds containers, starts the stack, verifies health, and tears down.
# Exit codes: 0 = success, 1 = failure
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
HEALTH_TIMEOUT=120  # seconds to wait for health checks
POLL_INTERVAL=5     # seconds between health polls

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cleanup() {
    log_info "Tearing down the stack..."
    docker compose -f "${COMPOSE_FILE}" down --volumes --remove-orphans 2>/dev/null || true
}

# Always clean up on exit
trap cleanup EXIT

# --------------------------------------------------------------------------
# Step 1: Build containers
# --------------------------------------------------------------------------
log_info "Building containers..."
if ! docker compose -f "${COMPOSE_FILE}" build; then
    log_error "Container build failed."
    exit 1
fi
log_info "Build completed successfully."

# --------------------------------------------------------------------------
# Step 2: Start the stack
# --------------------------------------------------------------------------
log_info "Starting the stack..."
if ! docker compose -f "${COMPOSE_FILE}" up -d; then
    log_error "Failed to start the stack."
    exit 1
fi
log_info "Stack started."

# --------------------------------------------------------------------------
# Step 3: Wait for health checks to pass
# --------------------------------------------------------------------------
log_info "Waiting for services to become healthy (timeout: ${HEALTH_TIMEOUT}s)..."

wait_for_healthy() {
    local service="$1"
    local elapsed=0

    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        local status
        status=$(docker compose -f "${COMPOSE_FILE}" ps --format json "$service" 2>/dev/null | \
                 python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('Health',''))" 2>/dev/null || echo "unknown")

        if [ "$status" = "healthy" ]; then
            log_info "  ${service} is healthy."
            return 0
        fi

        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
    done

    log_error "  ${service} did not become healthy within ${HEALTH_TIMEOUT}s."
    docker compose -f "${COMPOSE_FILE}" logs "$service" --tail=20
    return 1
}

SERVICES=("zookeeper" "kafka" "triton" "app")
for svc in "${SERVICES[@]}"; do
    if ! wait_for_healthy "$svc"; then
        log_error "Health check failed for service: ${svc}"
        exit 1
    fi
done

log_info "All services are healthy."

# --------------------------------------------------------------------------
# Step 4: Run basic endpoint tests
# --------------------------------------------------------------------------
log_info "Testing app /status endpoint..."
APP_RESPONSE=$(curl -sf http://localhost:5000/status 2>&1) || {
    log_error "App /status endpoint is not responding."
    log_error "Response: ${APP_RESPONSE}"
    exit 1
}
log_info "  App /status responded: ${APP_RESPONSE}"

log_info "Testing Triton /v2/health/ready endpoint..."
TRITON_RESPONSE=$(curl -sf http://localhost:8000/v2/health/ready 2>&1) || {
    log_error "Triton /v2/health/ready endpoint is not responding."
    log_error "Response: ${TRITON_RESPONSE}"
    exit 1
}
log_info "  Triton /v2/health/ready responded successfully."

# --------------------------------------------------------------------------
# Step 5: All checks passed
# --------------------------------------------------------------------------
log_info "========================================="
log_info "  Smoke test PASSED - all services OK"
log_info "========================================="
exit 0
