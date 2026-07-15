Install dependencies:

```bash
source ./.venv/bin/activate
pip install -r requirements.txt
```

Convert a notebook to a text file (and back):

```bash
jupytext --to py:percent notebook.ipynb      # notebook.ipynb -> notebook.py
jupytext --to notebook notebook.py           # notebook.py    -> notebook.ipynb
jupytext --to notebook notebook.py --update  # preserve existing outputs
```

Pair one notebook so both representations stay in sync on every save:

```bash
jupytext --set-formats ipynb,py:percent notebook.ipynb
```

Or pair every notebook in your project by adding a `jupytext.toml` file:

```
formats = "ipynb,py:percent"
```