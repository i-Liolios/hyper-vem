from setuptools import setup, find_packages

setup(
    name="hypervem",
    version="0.1",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "numpy>=1.23.5",
        "scipy>=1.17.0",
        "scikit-learn>=1.8.0",
        "pandas>=2.3.3",
        "matplotlib>=3.6.2",
        "seaborn>=0.13.2"
    ],
    extras_require={
        "kmodes": ["kmodes>=0.12.2"],
        "test": ["pytest>=9.0.2"]
    },
    python_requires = ">=3.11"
)