# Hermitian_Code

> **¿Primera vez con este proyecto?** Lee la **[Guía completa para principiantes](GUIA_COMPLETA.md)** — explica la física, la programación, cada archivo y cómo ejecutar todo paso a paso.

Implementación en Python de la formulación Hermitiana para estructuras de bandas fotónicas en metamateriales dispersivos, según:

> **A. Raman & S. Fan**, *Photonic Band Structure of Dispersive Metamaterials Formulated as a Hermitian Eigenvalue Problem*, Phys. Rev. Lett. **104**, 087401 (2010).

Framework modular con `scipy.sparse` que resuelve el problema de autovalores generalizado ωAx = Bx para materiales dispersivos tipo Drude/Lorentz.

## Física

El sistema acoplado (campos **H**, **E**, polarización **P**, velocidad **V**) se resuelve como:

$$\omega A x = B x, \qquad \hat{H} = A^{-1/2} B A^{-1/2}, \qquad \omega \hat{y} = \hat{H} \hat{y}$$

con modelo de Lorentz/Drude:

$$\varepsilon(\omega) = \varepsilon_\infty \left(1 + \frac{\omega_p^2}{\omega_0^2 - \omega^2 + i\gamma\omega}\right)$$

## Instalación

```bash
cd Hermitian_Code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Uso rápido

```bash
# Estructura de bandas TE — barras cuadradas plasmónicas (Fig. 1)
python -m src.main --mode TE --resolution 20 --nk 20 --nbands 15

# Caso con pérdidas + comparación perturbativa (Fig. 2)
python -m src.main --gamma 0.01 --perturb

# Ejemplos reproducibles
python examples/square_rods_TE.py
python examples/lossy_case.py
python examples/convergence_test.py
python examples/perturbation_comparison.py
```

## Tests

```bash
pytest tests/ -v
```

## Estructura

```
Hermitian_Code/
├── src/           # Módulos principales
├── examples/      # Casos del paper
├── tests/         # Pruebas unitarias
├── results/       # Figuras, bandas, logs, PDF
└── requirements.txt
```

## Parámetros del paper (Caso TE)

| Parámetro | Valor |
|-----------|-------|
| Geometría | Barras cuadradas, lado s = 0.25a |
| ε∞ (metal) | 1 |
| ω₀ | 0 (Drude) |
| ωp | 1 (unidades normalizadas) |
| Resolución | 20×20 |
| Γ (pérdidas) | 0.01 ωp |

## Unidades

Se adoptan unidades normalizadas: **a = 1** (red), **c = 1**, **μ₀ = 1**. Las frecuencias ω se reportan como ωc/a.

## Documentación científica

La documentación LaTeX está en `results/latex_doc/`. Para compilar:

```bash
cd results/latex_doc
pdflatex documentation.tex
bibtex documentation
pdflatex documentation.tex
pdflatex documentation.tex
```

## Referencia

Basado en Raman & Fan (2010). Incluye corrección del signo en la matriz B (modo TE) y en la corrección perturbativa (eq. 15).
