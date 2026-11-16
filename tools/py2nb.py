"""Convierte un script con celdas '# %%' / '# %% [markdown]' en un notebook .ipynb.
Uso: python tools/py2nb.py tools/src_nb01.py notebooks/01_Aproximaciones_y_Especificaciones.ipynb
"""
import sys
import nbformat as nbf


def convertir(src, dst):
    texto = open(src, encoding='utf-8').read()
    bloques = texto.split('# %%')
    nb = nbf.v4.new_notebook()
    for b in bloques:
        if not b.strip():
            continue
        if b.startswith(' [markdown]'):
            lineas = b.split('\n')[1:]
            md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in lineas).strip()
            nb.cells.append(nbf.v4.new_markdown_cell(md))
        else:
            nb.cells.append(nbf.v4.new_code_cell(b.split('\n', 1)[1].strip() if '\n' in b else ''))
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['colab'] = {'provenance': []}
    nbf.write(nb, dst)


if __name__ == '__main__':
    convertir(sys.argv[1], sys.argv[2])
