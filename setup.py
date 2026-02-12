from setuptools import setup, find_packages

setup(
    name="any-order-training",
    version="0.1.0",
    description="Any-Order Masked Training for Trajectory-Level Learning in LLM-Based Agents",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "minigrid>=2.3.0",
        "gymnasium>=0.29.0",
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
