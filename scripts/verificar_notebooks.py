"""Inspeção estática dos notebooks: proíbe caminhos fixos de ambiente.

Falha (exit ≠0) se encontrar ``../data`` ou ``/content`` em qualquer célula que
não seja a **primeira célula de código** (a célula de preparação) de cada
notebook. Garante que todos os caminhos venham de ``src.config``

Uso:
    python scripts/verificar_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PROIBIDOS = ("../data", "/content")


def celula_texto(cell: dict) -> str:
    """Retorna o código-fonte da célula como uma única string."""
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def inspecionar(caminho: Path) -> list[str]:
    """Inspeciona um notebook e retorna a lista de violações encontradas."""
    nb = json.loads(caminho.read_text(encoding="utf-8"))
    violacoes: list[str] = []
    primeira_code_vista = False
    for i, cell in enumerate(nb.get("cells", [])):
        eh_code = cell.get("cell_type") == "code"
        if eh_code and not primeira_code_vista:
            primeira_code_vista = True  # célula de preparação: ignorada
            continue
        texto = celula_texto(cell)
        for termo in PROIBIDOS:
            if termo in texto:
                violacoes.append(f"{caminho.name} célula #{i}: contém '{termo}'")
    return violacoes


def main() -> int:
    """Inspeciona todos os notebooks e retorna o exit code."""
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("AVISO: nenhum notebook encontrado em notebooks/")
        return 0

    todas: list[str] = []
    for nb in notebooks:
        todas.extend(inspecionar(nb))

    if todas:
        print("FALHA: caminhos fixos de ambiente fora da célula de preparação:")
        for v in todas:
            print(" -", v)
        return 1

    print(f"OK: {len(notebooks)} notebook(s) sem '../data' ou '/content' fora da preparação.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
