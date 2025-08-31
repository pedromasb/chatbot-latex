"""Setup script for ThesisChat package."""

from setuptools import setup, find_packages
import os
import pathlib

# Read the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements = []
    req_file = here / "requirements.txt"
    
    if req_file.exists():
        with open(req_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    # Handle conditional dependencies
                    if ";" not in line:
                        requirements.append(line)
                    else:
                        # For now, include all dependencies (can be refined)
                        req = line.split(";")[0].strip()
                        if req:
                            requirements.append(req)
    
    # Core requirements (fallback if requirements.txt is not found)
    if not requirements:
        requirements = [
            "sentence-transformers>=2.2.0",
            "pinecone-client>=2.2.0",
            "openai>=1.0.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "torch>=1.12.0",
            "transformers>=4.21.0"
        ]
    
    return requirements

# Read version from __init__.py
def get_version():
    """Extract version from __init__.py."""
    version_file = here / "thesis_chat" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"\'')
    return "1.0.0"

setup(
    # Package metadata
    name="thesis-chat",
    version=get_version(),
    description="LaTeX Document Processing and Conversational AI for Academic Documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/thesis-chat",
    author="ThesisChat Development Team",
    author_email="support@thesischat.com",
    
    # Classifiers help users find your project
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup :: LaTeX",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discoverability
    keywords="latex, academic, thesis, ai, nlp, embeddings, vector-search, rag, conversational-ai, research",
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies for different use cases
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "api": [
            "flask>=2.0.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
        ],
        "integration": [
            "requests>=2.25.0",
            "python-dotenv>=0.19.0",
        ],
        "analysis": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "wordcloud>=1.8.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "flask>=2.0.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "requests>=2.25.0",
            "python-dotenv>=0.19.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "wordcloud>=1.8.0",
        ]
    },
    
    # Package data
    package_data={
        "thesis_chat": [
            "data/*.json",
            "data/*.txt",
        ],
    },
    
    # Include additional files specified in MANIFEST.in
    include_package_data=True,
    
    # Entry points for command-line scripts (if needed in the future)
    entry_points={
        "console_scripts": [
            # "thesis-chat=thesis_chat.cli:main",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/thesis-chat/issues",
        "Source": "https://github.com/your-org/thesis-chat",
        "Documentation": "https://thesis-chat.readthedocs.io/",
        "Discussions": "https://github.com/your-org/thesis-chat/discussions",
    },
    
    # Zip safety
    zip_safe=False,
    
    # Platform compatibility
    platforms=["any"],
    
    # License information
    license="MIT",
    license_files=["LICENSE"],
    
    # Additional metadata for PyPI
    maintainer="ThesisChat Development Team",
    maintainer_email="support@thesischat.com",
)