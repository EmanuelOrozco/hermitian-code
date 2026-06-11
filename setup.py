"""Instalación del paquete Hermitian_Code."""

from setuptools import find_packages, setup

setup(
    name="hermitian-code",
    version="1.0.0",
    description="Estructura de bandas fotónicas dispersivas — formulación Hermitiana",
    author="Emanuel Cabrera Novoa",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "h5py>=3.8", "numba>=0.57"],
    },
    entry_points={
        "console_scripts": [
            "hermitian-bands=src.main:main",
        ],
    },
)
