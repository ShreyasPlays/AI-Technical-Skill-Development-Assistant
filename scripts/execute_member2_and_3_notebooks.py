from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def execute_notebook(nb_path: Path):
    print(f"Executing {nb_path.name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}}
    )
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved {nb_path.name} with live outputs!")

if __name__ == "__main__":
    # Execute Member 2 notebook
    nb2 = PROJECT_ROOT / "notebooks" / "02_eda_feature_engineering.ipynb"
    execute_notebook(nb2)

    # Execute Member 3 notebook
    nb3 = PROJECT_ROOT / "notebooks" / "03_baseline_models_evaluation.ipynb"
    execute_notebook(nb3)

    # Copy to alias 03_baseline_experiments.ipynb
    alias3 = PROJECT_ROOT / "notebooks" / "03_baseline_experiments.ipynb"
    with open(nb3, "r", encoding="utf-8") as f:
        nb3_content = nbformat.read(f, as_version=4)
    with open(alias3, "w", encoding="utf-8") as f:
        nbformat.write(nb3_content, f)
    print(f"Synchronized {alias3.name}")
