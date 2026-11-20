# Streamlit Community Cloud Deployment

Repository: `https://github.com/lemongood000/ML-Drug-Solubility`

## Streamlit settings

- Repository: `lemongood000/ML-Drug-Solubility`
- Branch: `main`
- Main file path: `app/app.py`
- Python version: 3.12
- Advanced settings: use `requirements-app.txt` if the dashboard allows a custom
  requirements file. Otherwise temporarily copy its contents into
  `requirements.txt` in the deployment branch.

## Required repository files

- `app/app.py`
- `src/`
- `artifacts/solubility_models.joblib`
- `requirements-app.txt`
- `.streamlit/config.toml`

The application only performs inference. It does not need the full training
dataset, PyTorch, test results, or intermediate OOF prediction files at runtime.

## Local launch

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m streamlit run app\app.py --server.port 8502
```

## Public URL

After deployment, Streamlit provides a public URL in the form:

```text
https://<app-name>.streamlit.app
```
