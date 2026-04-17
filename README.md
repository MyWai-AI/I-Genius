# Streamlit Template - MyWai Integration

Base template for Streamlit applications integrated with MyWai.

## Development Environment Setup

### 1. Prerequisites
```powershell
pip install uv
```

### 2. Project Setup
```powershell
# Clone and navigate to directory
cd vilma-agent

# Create virtual environment
uv venv
source .venv/bin/activate
```

### 3. Environment Configuration

Copy the template and configure variables:
```powershell
# Copy template
cp .env.template .env
```

Edit `.env` with your values:
```env
# Authentication mode
# true: Login with email/password in the interface
# false: Integration as toolkit with MyWai platform
DEBUG_MODE=false


# Execution environment  
# true: For local Docker with MyWai platform locally
# false: For production environment
LOCAL_HOST_RUN_ENV=false

# Azure DevOps credentials (required)
MYWAI_ARTIFACTS_MAIL=your_email@myw.ai
MYWAI_ARTIFACTS_TOKEN=your_azure_devops_token


# Debug mode credentials
MYWAI_USER=your_email@myw.ai
MYWAI_PASSWORD=your_password

# MyWai API endpoint
MYWAI_API_ENDPOINT=https://igenius.platform.myw.ai/api

```

### 4. Registry Authentication Setup

Before installing dependencies, configure access to Azure DevOps registry for the [mywai_python_integration_kit](https://dev.azure.com/zenatek-mywai/MYWAI/_git/PythonApiClient) package:

#### Windows PowerShell
```powershell
# Load variables from .env
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

# Create .netrc file for authentication
.\src\scripts\windows\create_netrc.ps1

# Install dependencies
uv sync
```

#### Linux/Mac
```bash
# Load variables from .env
export $(grep -v '^#' .env | xargs)

# Create .netrc file for authentication
bash src/scripts/linux/create_netrc.sh

# Install dependencies
uv sync
```

### 5. Local Execution
```powershell
# Activate environment
source .venv/bin/activate

# Start application
streamlit run main.py
```

## Execution Modes

### Debug Mode (`DEBUG_MODE=true`)
**To access the platform with email and password:**
- Manual login form in the interface
- Enter MyWai credentials directly
- Configurable endpoint in the form
- Logout button visible
- Expanded debug info

### Toolkit Mode (`DEBUG_MODE=false`)
**To connect to MyWai as integrated toolkit:**
- Authentication via `streamlit_post_message`
- Direct integration with MyWai platform
- No login form visible
- Production mode

### Local Host (`LOCAL_HOST_RUN_ENV=true`)
**To run local Docker image with MyWai platform locally:**
- Automatic Docker/localhost endpoint management
- Configuration for local development environment

### Production (`LOCAL_HOST_RUN_ENV=false`)
**For production environment:**
- Configuration for remote deployment

## Integration Guides

For detailed instructions on integrating your toolkit with MyWai, refer to the following guides:

- [Integration with MyWai 2.1](docs/1.%20Integration_with_mywai_2_1.md) - Guide for integrating with MyWai version 2.1
- [Integration with MyWai 3](docs/2.%20Integration_with_mywai_3.md) - Guide for integrating with MyWai version 3

These guides cover three main integration scenarios:
1. **Deployed Toolkit** - Using production URL
2. **Local Toolkit with MyWai on Cloud** - Using ngrok to expose local container port
3. **Local Toolkit with Local MyWai** - Using ngrok to expose local container port

## Architecture

```
src/streamlit_template/
├── auth/                    # MyWai Authentication
│   └── mywai_auth.py       # Login, payload, API initialization
├── core/                   # Configuration
│   └── config.py          # Environment variables, modes
└── ui/                     # User Interface
    ├── components.py       # Reusable components
    └── pages.py           # Page logic
```

## Adding New Features

### 1. New Pages
Modify `src/streamlit_template/ui/pages.py`:

```python
def create_my_page():
    """New custom page."""
    mywai_payload = get_mywai_payload()
    if mywai_payload is None:
        return
    
    initialize_mywai_apis(mywai_payload)
    
    st.header("My Page")
    # Add logic here
```

Call from `main.py` or add routing.

### 2. UI Components
Add to `src/streamlit_template/ui/components.py`:

```python
def render_my_component(data: dict):
    """Custom component."""
    st.subheader("My Component")
    st.json(data)
```

### 3. Business Logic Services
Create new module `src/streamlit_template/services/`:

```python
# src/streamlit_template/services/my_service.py
from mywai_python_integration_kit.apis.services.users_api import get_users

def process_users():
    """Custom business logic."""
    users = get_users()
    return [u for u in users if u.get('active')]
```

### 4. MyWai APIs
After `initialize_mywai_apis()`, use APIs from [mywai_python_integration_kit](https://dev.azure.com/zenatek-mywai/MYWAI/_git/PythonApiClient):

```python
from mywai_python_integration_kit.apis.services.users_api import get_users
from mywai_python_integration_kit.apis.services.projects_api import get_projects

# In your pages/components
users = get_users()
projects = get_projects()
```

### 5. Configuration
Add variables to `src/streamlit_template/core/config.py`:

```python
class Config:
    MY_SETTING = os.getenv("MY_SETTING", "default_value")
```

## Docker Build (Deploy Only)

Docker build is for deployment, not for local development:

#### Windows PowerShell
```powershell
# Use build script that handles environment variables
.\src\scripts\windows\docker_build.ps1
```

#### Linux/Mac
```bash
# Use build script that handles environment variables
bash src/scripts/linux/docker_build.sh
```

The script automatically reads from `.env` file and requires:
- `MYWAI_ARTIFACTS_MAIL`: Email for Azure DevOps registry
- `MYWAI_ARTIFACTS_TOKEN`: Token for Azure DevOps registry

Optional Docker variables:
- `DEBUG_MODE`: Debug mode in container (default: false)
- `LOCAL_HOST_RUN_ENV`: Local environment in container (default: false)

To run the container in local host and debug mode:
```
docker run -it --rm -p 2222:2222 -e LOCAL_HOST_RUN_ENV=true -e DEBUG_MODE=true streamlit-template
```
## Cloud deploy
To deploy the repository in cloud you use 2 files in the repo:
- `docker-compose.yml`
- `azure-pipeline-nextdev.yml`

To complete the deployment you can follow these two guides:
- Guide to deploy on server: 
[From Build to Registry to Server](https://dev.azure.com/zenatek-mywai/MYWAI/_wiki/wikis/MYWAI.wiki/115/Container-Deployment-Workflow-From-Build-to-Registry-to-Server).
- Guide to deploy on Cloud via Container Apps: 
[Container Apps Guide](https://dev.azure.com/zenatek-mywai/MYWAI/_wiki/wikis/MYWAI.wiki/118/Azure-Container-Apps)

## Repository Dependencies

- [mywai_python_integration_kit](https://dev.azure.com/zenatek-mywai/MYWAI/_git/PythonApiClient): MyWai APIs
- [streamlit_post_message](https://github.com/GitMarco27/streamlit_post_message): Platform communication


