# Guía completa del proyecto Hermitian_Code

**Para quién es este documento:** cualquier persona que quiera entender y usar este código, **sin necesidad de saber física avanzada ni programación previa**. Se explica desde cero qué hace el programa, cómo está organizado y cómo ejecutarlo paso a paso.

**Paper de referencia:** Raman & Fan, *Phys. Rev. Lett.* **104**, 087401 (2010).

---

## Tabla de contenidos

1. [¿Qué problema resuelve este código?](#1-qué-problema-resuelve-este-código)
2. [Conceptos de física en palabras simples](#2-conceptos-de-física-en-palabras-simples)
3. [Conceptos de programación que necesitas](#3-conceptos-de-programación-que-necesitas)
4. [Estructura del proyecto (mapa de carpetas)](#4-estructura-del-proyecto-mapa-de-carpetas)
5. [Explicación de cada archivo de código](#5-explicación-de-cada-archivo-de-código)
6. [Cómo instalar y ejecutar](#6-cómo-instalar-y-ejecutar)
7. [Flujo interno: qué hace el programa por dentro](#7-flujo-interno-qué-hace-el-programa-por-dentro)
8. [Parámetros que puedes cambiar](#8-parámetros-que-puedes-cambiar)
9. [Resultados: dónde están y cómo leerlos](#9-resultados-dónde-están-y-cómo-leerlos)
10. [Ejemplos incluidos](#10-ejemplos-incluidos)
11. [Tests (pruebas automáticas)](#11-tests-pruebas-automáticas)
12. [Estado de la revisión técnica](#12-estado-de-la-revisión-técnica)
13. [Glosario](#13-glosario)
14. [Preguntas frecuentes](#14-preguntas-frecuentes)

---

## 1. ¿Qué problema resuelve este código?

Imagina un material formado por **barras metálicas muy pequeñas** repetidas en un patrón periódico (como baldosas en un piso). La luz (ondas electromagnéticas) que viaja por ese material se comporta de forma especial: solo ciertas **frecuencias** pueden propagarse, y otras quedan bloqueadas.

Eso se representa con una **estructura de bandas**: un gráfico que muestra qué frecuencias están permitidas según la dirección de propagación de la luz.

El metal no tiene permitividad constante: depende de la frecuencia (es **dispersivo**). Eso hace el cálculo matemáticamente difícil. El paper de Raman y Fan muestra cómo convertirlo en un problema estándar de **autovalores** (buscar números especiales de una matriz), que es lo que este código resuelve en la computadora.

**En una frase:** este programa calcula numéricamente las frecuencias de luz permitidas en un cristal fotónico con inclusiones metálicas.

---

## 2. Conceptos de física en palabras simples

### Ondas electromagnéticas

La luz es una onda con dos campos acoplados:
- **E** (campo eléctrico)
- **H** (campo magnético)

Maxwell describe cómo se relacionan. En 2D hay dos polarizaciones:
- **TE:** el campo eléctrico está en el plano; el magnético sale de la pantalla (Hz en nuestro caso es al revés — Hz dominante en TE 2D del paper).
- **TM:** el campo magnético está en el plano; el eléctrico sale de la pantalla.

Este proyecto trabaja principalmente en **TE** (como la Fig. 1 del paper).

### Material dispersivo (Drude)

En un metal, la "respuesta" al campo eléctrico no es instantánea: los electrones libres oscilan. Eso se modela con la **frecuencia de plasma** ωp y, si hay fricción, un **amortiguamiento** γ.

En el código:
- **Metal:** ε∞ = 1, ωp = 1, ω₀ = 0 (modelo Drude sin pérdidas en el caso base).
- **Aire:** ε∞ = 1, sin polo activo (ωp = 0).

### Campos auxiliares V

Para que el problema sea lineal en la frecuencia ω, el paper introduce un campo extra **V** (velocidad de polarización). El vector de estado completo es:

```
x = (H, E, V)   →   en TE: (Hz, Ex, Ey, Vx, Vy)
```

### Estructura de bandas

Se calcula para distintos vectores de onda **k** (dirección de la luz en el cristal). En el paper se usa la ruta **Γ → X** (de k = 0 hasta k = π/a).

Cada punto del gráfico es una **frecuencia ω** de un modo permitido.

### Pérdidas

Si γ > 0, la frecuencia se vuelve **compleja**: ω = ω_real + i·ω_imag. La parte imaginaria indica cuánto se absorbe la luz. El código puede calcular eso de forma **exacta** o con **perturbación** (aproximación de primer orden).

---

## 3. Conceptos de programación que necesitas

### Python

Lenguaje de programación interpretado. Los archivos `.py` contienen instrucciones que Python ejecuta línea a línea.

### Entorno virtual (`.venv`)

Carpeta aislada con las librerías del proyecto. Evita conflictos con otros programas Python del sistema.

```bash
python3 -m venv .venv          # crear entorno
source .venv/bin/activate      # activarlo (Linux/Mac)
# En Windows: .venv\Scripts\activate
```

### Librerías usadas

| Librería | Para qué sirve |
|----------|----------------|
| **numpy** | Arrays y operaciones numéricas |
| **scipy** | Matrices dispersas y solver de autovalores |
| **matplotlib** | Gráficos |
| **pytest** | Ejecutar pruebas automáticas |

### Módulo y paquete

- Un **archivo** `.py` es un módulo (ej. `grid.py`).
- La carpeta `src/` con varios módulos es un **paquete**.
- Se importa así: `from src.grid import YeeGrid`

### Línea de comandos (terminal)

Texto que escribes para ejecutar programas:

```bash
python -m src.main --resolution 20
```

- `python -m src.main` → ejecuta `src/main.py`
- `--resolution 20` → argumento que modifica el comportamiento

### Matriz dispersa (sparse)

En lugar de guardar millones de ceros, solo se almacenan los valores distintos de cero. Esencial para problemas grandes en física computacional.

---

## 4. Estructura del proyecto (mapa de carpetas)

```
Hermitian_Code/
│
├── src/                    ← CÓDIGO PRINCIPAL (el cerebro del programa)
│   ├── main.py             ← Punto de entrada: lo que ejecutas desde terminal
│   ├── constants.py        ← Constantes físicas y configuración
│   ├── grid.py             ← Malla espacial (Yee grid)
│   ├── operators.py        ← Operadores matemáticos (matrices A, B, Ĥ)
│   ├── dispersive_materials.py  ← Modelo Drude/Lorentz
│   ├── hermitian_solver.py ← Resuelve el problema de autovalores
│   ├── eigenmodes.py       ← Extrae campos y energía de cada modo
│   ├── perturbation.py     ← Corrección de pérdidas (eq. 15)
│   ├── visualization.py    ← Genera gráficos
│   ├── validation.py       ← Comprueba hermiticidad y ortogonalidad
│   └── utils.py            ← Utilidades (logging, rutas k, etc.)
│
├── examples/               ← SCRIPTS DE EJEMPLO listos para ejecutar
│   ├── square_rods_TE.py   ← Caso del paper (Fig. 1)
│   ├── lossy_case.py       ← Pérdidas (Fig. 2)
│   ├── convergence_test.py ← Convergencia con la malla
│   └── perturbation_comparison.py
│
├── tests/                  ← PRUEBAS AUTOMÁTICAS (verifican que todo funcione)
│   ├── test_operators.py
│   ├── test_hermiticity.py
│   ├── test_eigenvalues.py
│   └── test_dispersion.py
│
├── results/                ← SALIDAS generadas al ejecutar
│   ├── figures/            ← Imágenes PNG y PDF
│   ├── band_structures/    ← Datos numéricos (.dat, .json)
│   ├── field_profiles/     ← Mapas de campos
│   ├── latex_doc/          ← Documentación científica en PDF
│   └── logs/               ← Registro de ejecución
│
├── requirements.txt        ← Lista de dependencias Python
├── setup.py                ← Instalación como paquete
├── README.md               ← Resumen técnico corto
└── GUIA_COMPLETA.md        ← Este documento
```

---

## 5. Explicación de cada archivo de código

### `src/constants.py`

Define **números y opciones fijas**:
- `MU0 = 1`, `LATTICE_A = 1` (unidades normalizadas)
- `Polarization.TE` / `Polarization.TM`
- `SimConfig`: dataclass con todos los parámetros de simulación (resolución, geometría, γ, etc.)

**Analogía:** es el "panel de configuración" del experimento virtual.

### `src/grid.py`

Construye la **malla de Yee**: divide la celda unitaria en N×N celdas y marca qué nodos son metal y cuáles aire.

Función principal: `YeeGrid.from_config(config)`

**Salida importante:** `metal_mask`, `eps_x`, `eps_y`, `omega_p_mask`.

### `src/dispersive_materials.py`

Describe matemáticamente el material:
- `DispersiveMaterial.drude()` → metal
- `DispersiveMaterial.vacuum()` → aire
- Soporta varios polos de Lorentz

### `src/operators.py`

El núcleo matemático. Construye las matrices:
- Derivadas finitas `Df_x`, `Db_x`, etc.
- Operadores rotacionales `CE`, `CH`
- Matriz de masa **A** (diagonal)
- Matriz dinámica **B**
- Hamiltoniano **Ĥ** = A⁻¹/² B A⁻¹/²

Función clave: `build_H_hat(grid, kx, ky)`

### `src/hermitian_solver.py`

Recibe Ĥ y encuentra sus **autovalores** (las frecuencias ω) con `scipy.sparse.linalg.eigsh`.

- `solve_hermitian()` → caso sin pérdidas (ω real)
- `solve_nonhermitian()` → caso con pérdidas (ω complejo)

### `src/eigenmodes.py`

Convierte el vector numérico `x` en campos físicos interpretables (Hz, Ex, Ey, Vx, Vy) y calcula la **energía** del modo.

### `src/perturbation.py`

Implementa la ecuación (15) del paper para estimar pérdidas sin resolver el sistema completo con γ.

### `src/visualization.py`

Crea gráficos con matplotlib:
- `plot_band_structure()` → diagrama de bandas
- `plot_field_intensity()` → mapas de |E|² y |V|²
- `plot_loss_comparison()` → exacto vs perturbativo

### `src/validation.py`

Herramientas de diagnóstico: ¿es Ĥ Hermitiana? ¿Los modos son ortogonales?

### `src/utils.py`

Funciones auxiliares: logging, ruta Γ→X, verificación de hermiticidad.

### `src/main.py`

**Orquesta todo:**
1. Lee argumentos de terminal
2. Crea la malla
3. Para cada punto k: ensambla Ĥ, resuelve, guarda bandas
4. Si hay pérdidas: compara métodos
5. Si se pide: exporta perfil de campo

---

## 6. Cómo instalar y ejecutar

### Paso 1: Abrir terminal en la carpeta del proyecto

```bash
cd Hermitian_Code
```

### Paso 2: Crear y activar entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar el programa principal

```bash
# Cálculo estándar de bandas (tarda ~1 min con N=20)
python -m src.main --mode TE --resolution 20 --nk 20 --nbands 15

# Versión rápida para probar (N=10, pocos puntos k)
python -m src.main --resolution 10 --nk 5 --nbands 5
```

### Paso 5: Ver resultados

Abre en tu explorador de archivos:
- `results/figures/band_structure.png`

---

## 7. Flujo interno: qué hace el programa por dentro

```
┌─────────────┐
│  Usuario    │  escribe: python -m src.main --resolution 20
└──────┬──────┘
       ▼
┌─────────────┐
│  main.py    │  lee argumentos → crea SimConfig
└──────┬──────┘
       ▼
┌─────────────┐
│  grid.py    │  construye malla N×N, marca metal/aire
└──────┬──────┘
       ▼
┌─────────────┐     Para cada k en Γ→X:
│ operators.py│  ensambla matriz Ĥ (sparse)
└──────┬──────┘
       ▼
┌──────────────────┐
│ hermitian_solver │  eigsh → encuentra ω₁, ω₂, … ωₙ
└──────┬───────────┘
       ▼
┌──────────────────┐
│ visualization.py │  guarda PNG en results/figures/
└──────────────────┘
```

**Ecuación que se resuelve en cada paso k:**

$$\omega \hat{y} = \hat{H} \hat{y}$$

donde $\hat{H} = A^{-1/2} B A^{-1/2}$ y $\hat{y} = A^{1/2} x$.

---

## 8. Parámetros que puedes cambiar

| Argumento | Significado | Valor por defecto | Efecto si lo aumentas |
|-----------|-------------|-------------------|------------------------|
| `--mode` | TE o TM | TE | Cambia qué campos se calculan |
| `--resolution` | N (celdas por lado) | 20 | Más preciso, más lento |
| `--fill-fraction` | Fracción de metal | 0.25 | Tamaño de las barras |
| `--shape` | square / circle | square | Forma de la inclusión |
| `--nk` | Puntos en ruta k | 20 | Más puntos en el gráfico |
| `--nbands` | Modos a calcular | 15 | Más curvas en el diagrama |
| `--gamma` | Amortiguamiento γ | 0 | Activa pérdidas si > 0 |
| `--perturb` | Comparar perturbación | false | Genera Fig. comparativa |
| `--sigma` | Desplazamiento solver | 0.3·ωp | Afecta qué modos encuentra |
| `--field-mode` | Índice del modo | -1 | Exporta mapa de campo |
| `--verbosity` | 0/1/2 | 1 | Más mensajes en pantalla |

---

## 9. Resultados: dónde están y cómo leerlos

### `results/figures/band_structure.png`

- **Eje X:** k normalizado (0 = Γ, 1 = X)
- **Eje Y:** frecuencia ω (unidades c/a)
- Cada línea es un **modo** distinto
- Bandas horizontales planas ≈ modos de plasmón superficial (como en el paper)

### `results/band_structures/bands.dat`

Archivo de texto con columnas: `k/π`, luego ω₁, ω₂, … ωₙ. Puedes abrirlo en Excel o Python.

### `results/figures/lossy_case_comparison.png`

Compara Im[ω] del método exacto vs perturbativo. Deben coincidir bien para γ pequeño.

### `results/logs/run.log`

Registro con hora y mensajes de cada ejecución. Útil si algo falla.

### `results/latex_doc/documentation.pdf`

Documento científico formal (para informe académico).

---

## 10. Ejemplos incluidos

| Script | Qué hace | Tiempo aprox. |
|--------|----------|---------------|
| `examples/square_rods_TE.py` | Reproduce Fig. 1 del paper | ~1 min |
| `examples/lossy_case.py` | Reproduce Fig. 2 (pérdidas) | ~10 s |
| `examples/convergence_test.py` | Error vs resolución N | ~2 min |
| `examples/perturbation_comparison.py` | Error perturbativo vs γ | ~30 s |

Ejecutar cualquiera:

```bash
python examples/square_rods_TE.py
```

---

## 11. Tests (pruebas automáticas)

Verifican que el código sea matemáticamente correcto:

```bash
pytest tests/ -v
```

**Qué comprueban:**
- La matriz B es Hermitiana
- Ĥ es Hermitiana
- El rotacional cumple C_H = C_E†
- Los autovalores son reales y positivos
- El modelo Drude da la ε(ω) esperada

**Estado actual:** 15 tests, todos pasan ✓

---

## 12. Estado de la revisión técnica

Revisión realizada el proyecto completo. Resumen:

| Aspecto | Estado | Notas |
|---------|--------|-------|
| Tests automáticos | ✓ 15/15 pasan | Hermiticidad, autovalores, dispersión |
| Hermiticidad de Ĥ (TE y TM) | ✓ Correcto | Corrección aplicada en bloque (E,Hz) |
| Solver de autovalores | ✓ Correcto | σ = 0.3·ωp evita modos espurios |
| Pérdidas perturbativas | ✓ Correcto | Coincide con exacto (~0% error) |
| Ejemplo Fig. 1 (bandas) | ✓ Genera gráfico | ω ≈ 0.38–0.48 en Γ |
| Ejemplo Fig. 2 (pérdidas) | ✓ Genera gráfico | Perturbación = exacto |
| CLI principal | ✓ Funciona | Probado con N=10 y N=20 |
| Documentación PDF | ✓ Compilada | `results/latex_doc/documentation.pdf` |

**Corrección importante respecto al código C++ original:** en el modo TE, el acoplamiento (E, Hz) de la matriz B debe usar **+i·CH** (no −i·CH) para que B sea Hermitiana. Esto ya está corregido en `operators.py`.

---

## 13. Glosario

| Término | Significado breve |
|---------|-------------------|
| **Autovalor** | Número λ tal que A·v = λ·v. Aquí λ = ω (frecuencia). |
| **Bloch** | Condición periódica en cristales: la onda adquiere una fase al cruzar la celda. |
| **Drude** | Modelo de metal: ε depende de ω con ω₀ = 0. |
| **eigsh** | Algoritmo de scipy para autovalores de matrices grandes y simétricas. |
| **Hermitiana** | Matriz igual a su conjugada transpuesta. Garantiza ω real. |
| **k** | Vector de onda (dirección y periodo espacial de la onda). |
| **Modo** | Patrón espacial de campo con una frecuencia específica. |
| **ω** | Frecuencia angular de la luz. |
| **ωp** | Frecuencia de plasma del metal. |
| **γ (gamma)** | Tasa de amortiguamiento (pérdidas). |
| **Yee grid** | Malla escalonada donde E y H viven en posiciones distintas. |
| **Sparse** | Matriz con mayoría de ceros; se guarda de forma eficiente. |
| **Shift-invert** | Técnica para encontrar autovalores cerca de un valor σ. |
| **TE / TM** | Dos polarizaciones de la luz en 2D. |

---

## 14. Preguntas frecuentes

### ¿Por qué tarda tanto con N=20?

El problema tiene (5 × N²) incógnitas en TE. Con N=20 son 2000 variables. El solver iterativo hace operaciones costosas en cada punto k. Usa N=10 para pruebas rápidas.

### ¿Qué es σ (sigma)?

Es el "centro de búsqueda" del solver. Con σ = 0.3 apuntamos al rango de frecuencias de las bandas plasmónicas. Si usas σ muy pequeño (0.01), el solver puede devolver modos no físicos (ω ≈ 0).

### ¿Puedo cambiar la forma de las barras?

Sí: `--shape circle` o modifica `grid.py` para otras geometrías.

### ¿Cómo exporto un mapa de campo?

```bash
python -m src.main --field-mode 0 --field-kindex 0
```

Guarda el modo 0 en k = Γ (primer punto de la ruta).

### ¿Dónde está el código C++ anterior?

Este repositorio es autónomo: no necesita otras carpetas del proyecto para ejecutarse.

### ¿Cómo cito este trabajo?

Raman & Fan, Phys. Rev. Lett. **104**, 087401 (2010), y menciona esta implementación académica.

---

## Siguiente paso recomendado

1. Activa el entorno virtual.
2. Ejecuta `python examples/square_rods_TE.py`.
3. Abre `results/figures/square_rods_TE_bands.png`.
4. Lee este documento sección por sección mientras miras el código en `src/`.

Si algo no funciona, revisa `results/logs/run.log` y ejecuta `pytest tests/ -v` para ver qué prueba falla.
