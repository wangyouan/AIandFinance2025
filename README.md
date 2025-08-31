# 人工智能金融学 (AI and Finance) — Course Repository

This repository contains the companion materials for the course “人工智能金融学 (AI and Finance)”. It includes Jupyter notebooks and scripts that demonstrate core AI/ML concepts and financial data workflows.

## Contents
- `lecture2_introduction_to_python.ipynb`
	- Unsupervised learning: KMeans clustering (sklearn)
	- A toy reinforcement learning simulation
	- Supervised learning: OLS, polynomial fits
	- Neural networks: sklearn MLP and Keras examples
- `lecture3_financial_data_acquisition_and_analysis.ipynb`
	- Accessing CSMAR via `csmarapi`
	- Accessing WRDS/CRSP via `wrds` to download monthly stock data and plotting
- `example_code.py`

## Prerequisites
- Python 3.9+ recommended
- VS Code with the Jupyter extension (or JupyterLab)

## Setup
Install base dependencies:

```powershell
pip install -r requirements.txt
```

Optional packages (used in Lecture 3):

```powershell
pip install wrds psycopg2-binary python-dotenv
# If you use CSMAR via Python, install the client package if available
# pip install csmarapi
```

## Credentials and Environment
- Create a `.env` file in the project root for CSMAR credentials:
	- `DB_USER=your_username`
	- `DB_PASS=your_password`
- For WRDS, you can set an environment variable to avoid username prompts:
	- `WRDS_USER=your_wrds_username`

## Using the Notebooks
1. Open the notebook in VS Code or Jupyter Lab.
2. Run cells top to bottom. For Lecture 3, ensure you have the optional packages and valid data access permissions (CSMAR/WRDS).

### PDF/HTML Export Tips
- Exporting to PDF via nbconvert requires Pandoc and a LaTeX distribution (e.g., MiKTeX on Windows).
	- Install Pandoc: https://pandoc.org/installing.html
	- Install TeX (for PDF): https://miktex.org/download
- HTML export generally works without TeX.

## Windows note for KMeans (MKL)
If you see a warning about a potential memory leak with KMeans on Windows/MKL, set the threads to 1 before importing scikit-learn:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
```

## Example: Running a Script
```powershell
python example_code.py
```

## Troubleshooting
- WRDS access errors: ensure your WRDS account has CRSP permissions.
- CSMAR login errors: confirm `.env` variables and network access.
- TensorFlow/Keras: CPU installation is sufficient for the demos.

---
© 2025 AI and Finance Course Materials