from setuptools import setup, find_packages

# Read the content of README.md as the detailed description of the project
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='credit_risk_detection',  # The name of the package
    version='0.1.0',  # Version number
    description='A package for credit card default risk analysis',
    long_description=long_description,  # Detailed description
    long_description_content_type='text/markdown',  # Specify the format of README
    author='Wenbo_Liu & Jiangao_Han',
    author_email='wbliu0528@gmail.com',
    packages=find_packages(),  # Automatically find packages (by default, the current directory and its subdirectories)
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'imbalanced-learn'
    ],
    python_requires='>=3.6',  # Python version requirement
    classifiers=[  # Classification information of the package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Using MIT License
        "Operating System :: OS Independent",
    ],
)
