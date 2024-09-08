from setuptools import setup, find_packages

# Read the contents of your README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="credit_risk_detection",  # Name of your package
    version="0.1.0",  # Version number
    author="Wenbo Liu & Jiangao Han",  # Author information
    author_email="wbliu0528@gmail.com",  # Author's email
    description="A package for credit card default risk analysis",  # A short description of your package
    long_description=long_description,  # Long description (from README)
    long_description_content_type="text/markdown",  # Format of the long description
    url="https://github.com/CrisisM/credit_risk_detection",  # URL to the project's homepage or repo (optional)
    packages=find_packages(),  # Automatically find and include all packages in the project
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "imbalanced-learn",
        "catboost",
        "xgboost",
        "lightgbm"
    ],  # Dependencies for your package
    python_requires='>=3.6',  # Minimum Python version required
    classifiers=[  # Additional metadata for your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # License type
        "Operating System :: OS Independent",
    ],
    license="MIT",  # The license under which your package is released
    keywords="credit card risk detection machine learning",  # Keywords to help users find your package
)
