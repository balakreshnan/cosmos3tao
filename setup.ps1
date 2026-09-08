# Re-creates the Python 3.14 virtual environment used by this repo (Windows / PowerShell).
# Training is NOT done locally; this env only generates data, submits TAO jobs, and runs inference clients.
$ErrorActionPreference = "Stop"

py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# nvidia-tao-client pulls nvidia-tao-core, whose full dependency tree includes uWSGI (Linux-only build).
# Install both without deps; the CLI only needs click/configparser/pyyaml/requests, already installed above.
.\.venv\Scripts\python.exe -m pip install --no-deps "nvidia-tao-client==6.26.3" "nvidia-tao-core==6.26.3"

.\.venv\Scripts\tao.exe --version
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
