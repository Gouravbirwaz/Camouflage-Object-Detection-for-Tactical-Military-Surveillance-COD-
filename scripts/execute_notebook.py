"""
Execute DGNet_Block_Visualization.ipynb cell-by-cell and embed outputs (stdout & matplotlib figures)
into the notebook JSON so it is fully pre-rendered.
"""

import io
import sys
import json
import base64
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_env = {"__name__": "__main__"}
    
    # Custom show hook to capture matplotlib figures
    captured_figures = []
    original_show = plt.show

    def captured_show(*args, **kwargs):
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('ascii')
            captured_figures.append(img_b64)
        plt.close('all')

    plt.show = captured_show

    execution_count = 1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        
        code = "".join(cell["source"])
        print(f"Executing Code Cell {execution_count} (index {i})...")
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        captured_figures.clear()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, global_env)
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            print(f"Error in cell {i}: {e}\n{tb_str}")
            cell["outputs"].append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": tb_str.splitlines(True)
            })
            break

        stdout_val = stdout_buf.getvalue()
        if stdout_val:
            cell["outputs"].append({
                "name": "stdout",
                "output_type": "stream",
                "text": stdout_val.splitlines(True)
            })

        for img_b64 in captured_figures:
            cell["outputs"].append({
                "output_type": "display_data",
                "data": {
                    "image/png": img_b64,
                    "text/plain": ["<Figure size>"]
                },
                "metadata": {}
            })

        execution_count += 1

    plt.show = original_show

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Successfully finished executing notebook and saved to {nb_path}!")

if __name__ == "__main__":
    notebook_file = Path("notebooks/DGNet_Block_Visualization.ipynb").resolve()
    run_notebook(notebook_file)
