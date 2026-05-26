"""setup.py for Memster — local-first long-term memory system."""

from setuptools import setup, find_packages

setup(
    name="memster",
    version="0.6.0",
    description="Local-first long-term memory system with PostgreSQL, hybrid retrieval, and MCP integration.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(
        include=["memster", "memster.*"],
        exclude=["benchmarks", "tests", "venv", "*.venv"],
    ),
    install_requires=[
        "psycopg2-binary>=2.9",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "nim": [
            "requests>=2.28",
        ],
        "mcp": [
            "mcp>=1.0",
        ],
        "all": [
            "psycopg2-binary>=2.9",
            "sentence-transformers>=2.2.0",
            "numpy>=1.24",
            "requests>=2.28",
            "mcp>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memster-mcp=memster_mcp_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
)