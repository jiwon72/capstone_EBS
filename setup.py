from setuptools import setup, find_packages

setup(
    name="trading_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "pytest-asyncio>=0.15.0",
        "httpx>=0.18.0",
        "python-dateutil>=2.8.0",
        "pydantic>=2.0.0",
        "newsapi-python>=0.2.7",
        "matplotlib>=3.8.2",
    ],
    python_requires=">=3.9",
) 