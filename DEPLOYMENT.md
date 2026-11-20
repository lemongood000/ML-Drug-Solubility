# Streamlit Community Cloud Deployment

Repository: `https://github.com/lemongood000/ML-Drug-Solubility`

## Streamlit settings

- Repository: `lemongood000/ML-Drug-Solubility`
- Branch: `main`
- Main file path: `app/app.py`
- Python version: 3.12
- Streamlit Community Cloud automatically installs the root `requirements.txt`.
  This file contains only inference dependencies and excludes PyTorch and
  training-time plotting packages.

## Required repository files

- `app/app.py`
- `src/`
- `artifacts/solubility_models.joblib`
- `requirements.txt`
- `.streamlit/config.toml`

The application only performs inference. It does not need the full training
dataset, PyTorch, test results, or intermediate OOF prediction files at runtime.

## Local launch

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\app.py --server.port 8502
```

## Public URL

After deployment, Streamlit provides a public URL in the form:

```text
https://<app-name>.streamlit.app
```
