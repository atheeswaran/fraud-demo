# Webinar 1 — ## Fraud Detection: From Notebook to Production System

Concepts demonstrated: ML model vs ML system · Pipe-and-filter pattern · Quality attributes  

---

## Backgroud

This demo starts with a 30-line notebook and grows it into a production-ready
ML service — one layer at a time. Each layer added corresponds to a box in the
Sculley "hidden technical debt" diagram shown in Session 2.

```
Step 1 — The notebook (what most students have)
Step 2 — MLflow experiment tracking (model versioning)
Step 3 — FastAPI inference service (serving infrastructure)
Step 4 — Pydantic schema validation (data validation)
Step 5 — Structured logging (monitoring & logging)
Step 6 — Config management (configuration management)
Step 7 — Pytest quality attribute tests (testing)
```

---

## Project Layout

```
fraud_demo/
├── README.md
├── .env                        # Config (step 6)
├── requirements.txt
│
├── data/
│   └── generate_data.py        # Generates synthetic fraud dataset
│
├── training/
│   ├── step1_notebook.py       # STEP 1: Raw notebook-style script
│   └── step2_mlflow_train.py   # STEP 2: Same script + MLflow tracking
│
├── app/
│   ├── config.py               # STEP 6: Config management
│   ├── schemas.py              # STEP 4: Pydantic input/output schemas
│   ├── pipeline.py             # Pipe-and-filter: 4 pure-function stages
│   ├── logger.py               # STEP 5: Structured logging setup
│   └── main.py                 # STEP 3: FastAPI app
│
└── tests/
    └── test_quality_attrs.py   # STEP 7: Tests that encode quality attributes
```

---


### Setup and Implementation
# Create a virtual environment inside the project
python -m venv venv
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# creates data/fraud_data.csv
python data/generate_data.py    

# trains model, logs to MLflow
python training/step2_mlflow_train.py 

# open http://127.0.0.1:5000 in browser
mlflow ui                              


### Note:

In `training/step1_notebook.py` - no schema validation, no error handling, no logging,
  hardcoded paths, no versioning, no serving.

In `training/step2_mlflow_train.py`. 
- `mlflow.start_run()` wraps the training
- `mlflow.log_params()` records hyperparameters
- `mlflow.sklearn.log_model()` versions the artifact

Start the server:
```bash
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs. Try a live prediction.
Show a bad request — watch Pydantic reject it with a structured error.
Show the logs in the terminal — structured JSON output.

In `app/pipeline.py`.
`validate_input → extract_features → run_model → format_response`
IS the pipe-and-filter pattern in Python.

Run the tests:
```bash
pytest tests/ -v
```
Take-home challenge:
"Add a /health endpoint that returns model version and last training timestamp."





## Changing Python Version for ML Projects on Windows

Most ML libraries (scikit-learn, MLflow, TensorFlow) publish **pre-built wheels** for stable Python versions. When you're on the latest Python release, those wheels often don't exist yet — and pip falls back to compiling from source, which requires a C compiler. On Windows, that compiler isn't installed by default, so you get errors like `Unknown compiler: gcc, clang, cl`.

The solution is to run a stable Python version **alongside** your default one. Windows supports this cleanly.

---

### Step 1 — Download the right Python version

Go to **https://www.python.org/downloads/** and download the version you need (Python 3.11 is recommended for the current ML ecosystem).

On the downloads page, scroll to **Files** at the bottom and pick:
```
Windows installer (64-bit)   →   python-3.11.x-amd64.exe
```
Avoid `.tar.gz` or `.zip` files — those are source code, not installers.

---

### Step 2 — Install it alongside your existing Python

Run the `.exe`. On the first screen:
- ✅ Check **"Add python.exe to PATH"**
- Click **"Customize installation"** → keep all defaults → **Install**

Your existing Python version is not affected. Windows keeps both.

---

### Step 3 — Verify both versions are available

Open a terminal and run:
```powershell
python --version      # your default (e.g. 3.14)
py -3.11 --version    # the newly installed one
```

The `py` launcher lets you call any installed version by number.

---

### Step 4 — Create a project venv with the specific version

Always create your virtual environment explicitly with the version you need:
```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python --version      # should now show 3.11.x inside the venv
```

Once activated, everything inside the venv — `python`, `pip`, installed packages — uses 3.11, regardless of your system default.

---

### Step 5 — Install packages normally

```powershell
pip install -r requirements.txt
```

With a 3.11 venv active, pip finds pre-built wheels for all major ML libraries and installs without needing a compiler.

---

### Key concept — virtual environments are version-locked

The Python version is baked into the venv at creation time. If you ever see compiler errors again, the quickest check is:

```powershell
python --version
```

Run this **after** activating the venv. If it shows an unsupported version, deactivate, delete the venv folder, and recreate it with `py -3.11 -m venv venv`.

---

### Quick reference

| Task                              | Command                 |
|-----------------------------------|-------------------------|
| Check all installed versions      | `py --list`             |
| Create venv with specific version | `py -3.11 -m venv venv` |
| Activate venv (Windows)           | `venv\Scripts\activate` |
| Confirm active Python version     | `python --version`      |
| Deactivate venv                   | `deactivate`            |