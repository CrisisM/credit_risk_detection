from setuptools import setup, find_packages

setup(
    name='credit_risk_detection',  # 包的名字
    version='0.1.0',  # 版本号
    description='A package for credit card default risk analysis',
    author='Wenbo_Liu & Jiangao_Han',
    author_email='wbliu0528@gmail.com',
    packages=find_packages(),  # 自动找到 package
    install_requires=[
        'numpy',  # 依赖项
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'imbalanced-learn'
    ],
    python_requires='>=3.6',  # Python 版本要求
)
