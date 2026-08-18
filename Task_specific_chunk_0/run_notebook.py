import json
import traceback
from pathlib import Path


notebook_path = Path("Day1_Task1_Document_Ingestion.ipynb")
nb = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}

for index, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    code = "".join(cell.get("source", []))
    if not code.strip():
        continue
    print(f"\n--- Running code cell {index} ---")
    try:
        exec(compile(code, f"{notebook_path.name}:cell-{index}", "exec"), namespace)
    except Exception:
        print(f"\nFAILED in code cell {index}")
        traceback.print_exc()
        raise SystemExit(1)

print("\nNotebook executed successfully.")
