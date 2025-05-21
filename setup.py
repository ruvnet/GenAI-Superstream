"""
Setup script for the GenAI-Superstream package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="genai-superstream",
    version="0.1.0",
    description="Machine learning model exposed through a Model Context Protocol (MCP) server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GenAI Superstream Team",
    author_email="example@example.com",
    url="https://github.com/yourusername/GenAI-Superstream",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'genai-superstream=main:main',
        ],
    },
)