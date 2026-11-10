# Producto de difusión (course article)

Article for the course deliverable "Producto de difusión" (Team 8). Every number comes from the graded
notebooks `notebooks/Avance1` to `Avance5`.

- `articulo.tex`, `referencias.bib`, `logo.jpeg`, `figuras/`: sources.
- `articulo.pdf`: built article.
- `figure_scripts/`: scripts that draw the figures.

Build with pdfLaTeX and Biber (on Overleaf: compiler pdfLaTeX, bibliography tool Biber):

```sh
pdflatex articulo.tex && biber articulo && pdflatex articulo.tex && pdflatex articulo.tex
```

Redraw the figures from the repository root:

```sh
python reports/articulo-difusion/figure_scripts/roc_budget.py . reports/articulo-difusion/figuras/fig_roc_budget.png
python reports/articulo-difusion/figure_scripts/beat_and_map.py reports/articulo-difusion/figuras
```

`roc_budget.py` reads `outputs/avance5_final/sph_predictions.npz` and
`reports/avance5_final/presupuestos_derivacion.csv`, which the Avance 5 notebook writes.
