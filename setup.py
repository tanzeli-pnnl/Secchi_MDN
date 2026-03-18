#!/usr/bin/env python3
"""
Setup script for the Python project.
"""

from setuptools import setup, find_packages

setup(
    name="secchi-mdn-reproduction",
    version="0.1.0",
    description="Reproduction workflow for Maciel et al. (2023) Secchi MDN models",
    author="OpenAI Codex",
    author_email="support@openai.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "joblib>=1.3",
        "numpy>=1.24",
        "openpyxl>=3.1",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "torch>=2.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
