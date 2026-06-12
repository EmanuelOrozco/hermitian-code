"""Package installation for Hermitian Code."""

from setuptools import find_packages, setup

setup(
    name="hermitian-code",
    version="1.0.0",
    description="Hermitian eigenvalue formulation for dispersive photonic bands",
    author="Emanuel Cabrera Novoa",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "hermitian-bands=src.main:main",
        ],
    },
)
