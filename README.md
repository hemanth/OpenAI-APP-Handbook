# openai-apps-handbook

This guide walks you through creating your app for chatGPT using Apps SDK.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Step-by-Step: Create Your App](#step-by-step-create-your-app)
4. [Adding Custom Widgets](#adding-custom-widgets)
5. [Adding New Tools](#adding-new-tools)
6. [Input Validation](#input-validation)
7. [Testing Your Server](#testing-your-server)
8. [Deployment Considerations](#deployment-considerations)
9. [Common Patterns](#common-patterns)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.10+** installed
- Basic understanding of:
  - Python dataclasses and type hints
  - FastAPI/async Python
  - HTTP/REST concepts
  - HTML (for widget templates)
- **Optional**: MCP Inspector for testing

---

## Understanding the Architecture

An MCP server app has three core components:

### 1. Widget Definitions
Widgets are UI components that render in ChatGPT. Each widget needs:
- **HTML template**: The UI structure (often loading external JS/CSS)
- **Metadata**: OpenAI-specific hints for rendering
- **Template URI**: Unique identifier (e.g., `ui://widget/my-widget.html`)

### 2. MCP Protocol Handlers
Functions that respond to ChatGPT's requests:
- `list_tools()`: Register available tools
- `list_resources()`: Expose widgets as resources
- `call_tool_request()`: Execute tool logic
- `read_resource()`: Serve widget HTML

### 3. Transport Layer
FastAPI + Uvicorn serving:
- `GET /mcp`: SSE stream for protocol communication
- `POST /mcp/messages`: Follow-up messages for sessions

---

## Step-by-Step: Create Your App

### Step 1: Set Up Your Project

```bash
# Create project directory
mkdir myChatGPTApp
cd myChatGPTApp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << EOF
mcp[fastapi]>=0.1.0
fastapi>=0.115.0
uvicorn>=0.30.0
EOF

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Create Your Main Application File

Create `main.py` with the basic structure:

```python
"""My Custom MCP Server"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Define your widget data structure
@dataclass(frozen=True)
class MyWidget:
    identifier: str
    title: str
    template_uri: str
    invoking: str
    invoked: str
    html: str
    response_text: str

# Initialize FastMCP
mcp = FastMCP(
    name="my-mcp-app",
    sse_path="/mcp",
    message_path="/mcp/messages",
    stateless_http=True,
)

# Your widgets will go here
widgets: List[MyWidget] = []

# Helper dictionaries for lookups
WIDGETS_BY_ID: Dict[str, MyWidget] = {}
WIDGETS_BY_URI: Dict[str, MyWidget] = {}

# MIME type for widget HTML
MIME_TYPE = "text/html+skybridge"
```

### Step 3: Define Your Widgets

Add widget definitions to your `widgets` list:

```python
widgets: List[MyWidget] = [
    MyWidget(
        identifier="my-first-tool",
        title="My First Tool",
        template_uri="ui://widget/my-first-tool.html",
        invoking="Processing your request",
        invoked="Request completed",
        html=(
            "<div id=\"my-widget-root\"></div>\n"
            "<link rel=\"stylesheet\" href=\"https://your-cdn.com/styles.css\">\n"
            "<script type=\"module\" src=\"https://your-cdn.com/app.js\"></script>"
        ),
        response_text="Tool executed successfully!",
    ),
]

# Build lookup dictionaries
WIDGETS_BY_ID = {w.identifier: w for w in widgets}
WIDGETS_BY_URI = {w.template_uri: w for w in widgets}
```

### Step 4: Define Input Schema

Create a Pydantic model for input validation:

```python
class MyToolInput(BaseModel):
    """Schema for tool inputs."""

    # Use alias for camelCase API but snake_case Python
    user_query: str = Field(
        ...,
        alias="userQuery",
        description="The user's input query",
    )

    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

# JSON Schema for MCP protocol
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "userQuery": {
            "type": "string",
            "description": "The user's input query",
        },
        "options": {
            "type": "object",
            "description": "Optional parameters",
        }
    },
    "required": ["userQuery"],
    "additionalProperties": False,
}
```

### Step 5: Implement MCP Handlers

```python
def _tool_meta(widget: MyWidget) -> Dict[str, Any]:
    """Generate OpenAI-specific metadata for widgets."""
    return {
        "openai/outputTemplate": widget.template_uri,
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
        "openai/widgetAccessible": True,
        "openai/resultCanProduceWidget": True,
        "annotations": {
            "destructiveHint": False,
            "openWorldHint": False,
            "readOnlyHint": True,
        }
    }

@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    """Register all available tools."""
    return [
        types.Tool(
            name=widget.identifier,
            title=widget.title,
            description=widget.title,
            inputSchema=deepcopy(TOOL_INPUT_SCHEMA),
            _meta=_tool_meta(widget),
        )
        for widget in widgets
    ]

@mcp._mcp_server.list_resources()
async def _list_resources() -> List[types.Resource]:
    """Expose widgets as resources."""
    return [
        types.Resource(
            name=widget.title,
            title=widget.title,
            uri=widget.template_uri,
            description=f"{widget.title} widget markup",
            mimeType=MIME_TYPE,
            _meta=_tool_meta(widget),
        )
        for widget in widgets
    ]

@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> List[types.ResourceTemplate]:
    """Define resource templates."""
    return [
        types.ResourceTemplate(
            name=widget.title,
            title=widget.title,
            uriTemplate=widget.template_uri,
            description=f"{widget.title} widget markup",
            mimeType=MIME_TYPE,
            _meta=_tool_meta(widget),
        )
        for widget in widgets
    ]

async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    """Serve widget HTML when requested."""
    widget = WIDGETS_BY_URI.get(str(req.params.uri))
    if widget is None:
        return types.ServerResult(
            types.ReadResourceResult(
                contents=[],
                _meta={"error": f"Unknown resource: {req.params.uri}"},
            )
        )

    contents = [
        types.TextResourceContents(
            uri=widget.template_uri,
            mimeType=MIME_TYPE,
            text=widget.html,
            _meta=_tool_meta(widget),
        )
    ]

    return types.ServerResult(types.ReadResourceResult(contents=contents))

async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    """Execute tool logic and return results."""
    widget = WIDGETS_BY_ID.get(req.params.name)
    if widget is None:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Unknown tool: {req.params.name}",
                    )
                ],
                isError=True,
            )
        )

    # Validate input
    arguments = req.params.arguments or {}
    try:
        payload = MyToolInput.model_validate(arguments)
    except ValidationError as exc:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Input validation error: {exc.errors()}",
                    )
                ],
                isError=True,
            )
        )

    # YOUR BUSINESS LOGIC GOES HERE
    # Example: process payload.user_query and payload.options
    result_data = {
        "query": payload.user_query,
        "processed": True,
    }

    # Build embedded widget resource
    widget_resource = types.EmbeddedResource(
        type="resource",
        resource=types.TextResourceContents(
            uri=widget.template_uri,
            mimeType=MIME_TYPE,
            text=widget.html,
            title=widget.title,
        ),
    )

    meta: Dict[str, Any] = {
        "openai.com/widget": widget_resource.model_dump(mode="json"),
        "openai/outputTemplate": widget.template_uri,
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
        "openai/widgetAccessible": True,
        "openai/resultCanProduceWidget": True,
    }

    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=widget.response_text,
                )
            ],
            structuredContent=result_data,
            _meta=meta,
        )
    )

# Register request handlers
mcp._mcp_server.request_handlers[types.CallToolRequest] = _call_tool_request
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource
```

### Step 6: Create FastAPI App with CORS

```python
app = mcp.streamable_http_app()

# Add CORS for local testing
try:
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
except Exception:
    pass

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
```

### Step 7: Run Your Server

```bash
python main.py
```

Your server is now running at `http://localhost:8000`!

---

## Adding Custom Widgets

### Option 1: External CDN Assets (Recommended)

If your UI is a JavaScript bundle hosted on a CDN:

```python
MyWidget(
    identifier="my-dashboard",
    title="Analytics Dashboard",
    template_uri="ui://widget/dashboard.html",
    invoking="Loading dashboard",
    invoked="Dashboard loaded",
    html=(
        "<div id=\"dashboard-root\"></div>\n"
        "<link rel=\"stylesheet\" href=\"https://cdn.example.com/dashboard.css\">\n"
        "<script type=\"module\" src=\"https://cdn.example.com/dashboard.js\"></script>"
    ),
    response_text="Dashboard rendered successfully",
)
```

### Option 2: Inline HTML

For simple static widgets:

```python
MyWidget(
    identifier="simple-card",
    title="Info Card",
    template_uri="ui://widget/card.html",
    invoking="Creating card",
    invoked="Card created",
    html=(
        "<div style='padding: 20px; border: 1px solid #ccc;'>"
        "  <h2>Hello from MCP!</h2>"
        "  <p>This is a simple inline widget.</p>"
        "</div>"
    ),
    response_text="Info card displayed",
)
```

### Option 3: Local Static Files

Serve local HTML/JS/CSS files via FastAPI static file mounting:

```python
from fastapi.staticfiles import StaticFiles

# Add this before creating the MCP app
app.mount("/static", StaticFiles(directory="static"), name="static")

# Reference in widget
html=(
    "<div id=\"my-root\"></div>\n"
    "<link rel=\"stylesheet\" href=\"/static/my-widget.css\">\n"
    "<script type=\"module\" src=\"/static/my-widget.js\"></script>"
)
```

---

## Adding New Tools

### Pattern 1: One Tool Per Widget (Simple)

Each tool maps directly to a widget:

```python
widgets = [
    MyWidget(identifier="search", title="Search", ...),
    MyWidget(identifier="filter", title="Filter", ...),
    MyWidget(identifier="export", title="Export", ...),
]
```

### Pattern 2: Multiple Tools, Shared Widget (Advanced)

Different tools can render the same widget with different data:

```python
# In _call_tool_request():
if req.params.name == "search-users":
    result_data = {"type": "user", "results": [...]}
elif req.params.name == "search-products":
    result_data = {"type": "product", "results": [...]}

# Both use the same "search-results" widget
widget = WIDGETS_BY_ID["search-results"]
```

### Pattern 3: Dynamic Widget Selection

Choose widget based on input:

```python
async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    payload = MyToolInput.model_validate(req.params.arguments or {})

    # Select widget based on input type
    if payload.output_format == "table":
        widget = WIDGETS_BY_ID["table-view"]
    elif payload.output_format == "chart":
        widget = WIDGETS_BY_ID["chart-view"]
    else:
        widget = WIDGETS_BY_ID["list-view"]

    # Proceed with widget...
```

---

## Input Validation

### Basic Validation

```python
class BasicInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)
```

### Complex Validation with Custom Validators

```python
from pydantic import field_validator

class AdvancedInput(BaseModel):
    email: str
    age: int

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()

    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v
```

### Nested Objects

```python
class FilterOptions(BaseModel):
    category: str
    min_price: float = 0.0
    max_price: float = 1000.0

class SearchInput(BaseModel):
    query: str
    filters: FilterOptions = Field(default_factory=FilterOptions)
```

---

## Testing Your Server

### 1. Manual Testing with curl

```bash
# List available tools
curl http://localhost:8000/mcp

# Call a tool (via SSE stream, more complex - use MCP Inspector instead)
```

### 2. Use MCP Inspector

Install and run the MCP Inspector:

```bash
npm install -g @modelcontextprotocol/inspector
mcp-inspector
```

Then connect to `http://localhost:8000/mcp`.

### 3. Unit Tests

Create `test_main.py`:

```python
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_list_tools():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/mcp")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_tool_validation():
    # Test your input validation logic
    from main import MyToolInput

    valid_input = {"userQuery": "test"}
    payload = MyToolInput.model_validate(valid_input)
    assert payload.user_query == "test"

    invalid_input = {"wrongField": "test"}
    with pytest.raises(Exception):
        MyToolInput.model_validate(invalid_input)
```

Run tests:

```bash
pip install pytest pytest-asyncio httpx
pytest test_main.py
```

---

## Deployment Considerations

### Environment Variables

```python
import os

PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
    )
```

### Production Configuration

```python
# Use gunicorn for production
# requirements.txt
gunicorn>=21.0.0

# Command to run
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["python", "main.py"]
```

Build and run:

```bash
docker build -t my-mcp-app .
docker run -p 8000:8000 my-mcp-app
```

### Security

```python
# 1. Add authentication middleware
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid token")
    return credentials

# Apply to specific routes or globally

# 2. Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/mcp")
@limiter.limit("100/minute")
async def mcp_endpoint(request: Request):
    ...
```

---

## Common Patterns

### Pattern: Database Integration

```python
import asyncpg

# Initialize DB pool
DB_POOL = None

async def get_db_pool():
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = await asyncpg.create_pool(
            "postgresql://user:pass@localhost/dbname"
        )
    return DB_POOL

# In your tool handler
async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch("SELECT * FROM items WHERE ...")

    result_data = {"items": [dict(r) for r in results]}
    # Return with widget...
```

### Pattern: External API Calls

```python
import httpx

async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    payload = MyToolInput.model_validate(req.params.arguments or {})

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.example.com/search",
            params={"q": payload.user_query}
        )
        api_data = response.json()

    result_data = {"results": api_data}
    # Return with widget...
```

### Pattern: File Processing

```python
import aiofiles

async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    # Read uploaded file (if you add file upload endpoint)
    async with aiofiles.open(file_path, mode='r') as f:
        content = await f.read()

    # Process content...
    processed = process_data(content)

    result_data = {"processed": processed}
    # Return with widget...
```

### Pattern: Caching

```python
from functools import lru_cache
import asyncio

# Sync cache
@lru_cache(maxsize=128)
def expensive_computation(param: str) -> dict:
    # Expensive operation
    return {"result": ...}

# Async cache (use aiocache)
from aiocache import cached

@cached(ttl=300)  # 5 minutes
async def fetch_data(key: str) -> dict:
    # Expensive async operation
    return {"data": ...}
```

---

## Troubleshooting

### Issue: Tools Not Appearing in ChatGPT

**Check:**
1. Server is running: `curl http://localhost:8000/mcp`
2. Tools are registered: Verify `_list_tools()` returns your tools
3. Input schema is valid JSON Schema
4. Metadata includes `openai/*` fields

### Issue: Widget Not Rendering

**Check:**
1. `template_uri` matches between widget and metadata
2. HTML is valid and includes root element
3. External CSS/JS URLs are accessible
4. MIME type is `text/html+skybridge`
5. `_meta` includes `openai/widgetAccessible: true`

### Issue: Input Validation Errors

**Check:**
1. Field names match between schema and Pydantic model (use `alias`)
2. Required fields are marked with `...` in Pydantic
3. JSON schema `required` array matches Pydantic required fields
4. Test validation independently:

```python
# Test in Python console
from main import MyToolInput

test_input = {"userQuery": "test"}
result = MyToolInput.model_validate(test_input)
print(result)
```

### Issue: CORS Errors

**Fix:**
Ensure CORS middleware is properly configured:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
```

### Issue: Server Crashes on Startup

**Check:**
1. All imports are available: `pip list`
2. Port 8000 is not already in use: `lsof -i :8000` (macOS/Linux)
3. Virtual environment is activated
4. Python version is 3.10+: `python --version`

---

## Next Steps

1. **Customize widgets**: Replace pizza examples with your domain
2. **Add real data**: Connect to databases, APIs, or file systems
3. **Implement auth**: Add authentication/authorization
4. **Add logging**: Use Python's `logging` module
5. **Monitor performance**: Add metrics and tracing
6. **Write tests**: Achieve >80% code coverage
7. **Deploy**: Use Docker, cloud platforms, or serverless

---

## Additional Resources

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAI Apps SDK Examples](https://github.com/openai/openai-apps-sdk-examples)

---

## Support

For issues with:
- **MCP Protocol**: Check MCP SDK documentation
- **FastAPI**: FastAPI community forums
- **This template**: Open an issue in the repository

Happy building!
