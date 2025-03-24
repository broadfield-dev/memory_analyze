from setuptools import setup, find_packages

setup(
    name="memory_analyze",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub",
        "google-generativeai",
        "groq"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An optional LLM-based analysis module for memory_core",
    url="https://github.com/username/memory_analyze",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
