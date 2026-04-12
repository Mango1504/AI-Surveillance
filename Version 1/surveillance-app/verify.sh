#!/bin/bash
# Verification script to check if the AI Surveillance System is properly configured

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   AI Surveillance System - Environment Verification Script     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

# Helper functions
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        PASSED=$((PASSED+1))
    else
        echo -e "${RED}✗${NC} $1 is NOT installed"
        FAILED=$((FAILED+1))
    fi
}

check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Port $1 is active"
        PASSED=$((PASSED+1))
    else
        echo -e "${YELLOW}○${NC} Port $1 is not active (start your services)"
        FAILED=$((FAILED+1))
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} File found: $1"
        PASSED=$((PASSED+1))
    else
        echo -e "${RED}✗${NC} File missing: $1"
        FAILED=$((FAILED+1))
    fi
}

check_endpoint() {
    if curl -s -f $1 > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Endpoint accessible: $1"
        PASSED=$((PASSED+1))
    else
        echo -e "${RED}✗${NC} Endpoint not accessible: $1"
        FAILED=$((FAILED+1))
    fi
}

# Start checks
echo -e "${BLUE}1. Checking Required Software${NC}"
echo "─────────────────────────────────────────────────────────────────"
check_command "node"
check_command "npm"
check_command "python"
echo ""

echo -e "${BLUE}2. Checking Project Files${NC}"
echo "─────────────────────────────────────────────────────────────────"
check_file "package.json"
check_file "src/App.jsx"
check_file "src/pages/Home.jsx"
check_file "second_CORS_enabled.py"
echo ""

echo -e "${BLUE}3. Checking Node Modules${NC}"
echo "─────────────────────────────────────────────────────────────────"
if [ -d "node_modules" ]; then
    echo -e "${GREEN}✓${NC} node_modules directory exists"
    PASSED=$((PASSED+1))
else
    echo -e "${YELLOW}○${NC} node_modules not found - run 'npm install'"
    FAILED=$((FAILED+1))
fi
echo ""

echo -e "${BLUE}4. Checking Active Services${NC}"
echo "─────────────────────────────────────────────────────────────────"
check_port 5000
check_port 3000
echo ""

echo -e "${BLUE}5. Checking API Endpoints (requires backend running)${NC}"
echo "─────────────────────────────────────────────────────────────────"
check_endpoint "http://localhost:5000/status"
check_endpoint "http://localhost:5000/grid-info"
check_endpoint "http://localhost:5000/stream"
echo ""

# Summary
echo -e "${BLUE}═════════════════════════════════════════════════════════════════${NC}"
echo -e "Summary: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! System is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Backend: python second.py"
    echo "  2. Frontend: npm run dev"
    echo "  3. Open: http://localhost:3000"
    exit 0
else
    echo -e "${YELLOW}⚠ Some checks failed. Please review the errors above.${NC}"
    echo ""
    echo "Common issues:"
    echo "  • Node/npm not installed: Install Node.js from nodejs.org"
    echo "  • Python not installed: Install Python 3.8+ from python.org"
    echo "  • node_modules missing: Run 'npm install' in this directory"
    echo "  • Services not running: Start Flask backend and React dev server"
    echo "  • Ports in use: Check if 3000 or 5000 are already in use"
    exit 1
fi
