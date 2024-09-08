from setuptools import setup, find_packages

# 读取 README.md 文件中的内容作为项目的详细描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='credit_risk_detection',  # 包的名字
    version='0.1.0',  # 版本号
    description='A package for credit card default risk analysis',
    long_description=long_description,  # 详细描述
    long_description_content_type='text/markdown',  # 指定README的格式
    author='Wenbo_Liu & Jiangao_Han',
    author_email='wbliu0528@gmail.com',
    packages=find_packages(),  # 自动找到 package（默认当前目录及其子目录）
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'imbalanced-learn'
    ],
    python_requires='>=3.6',  # Python 版本要求
    classifiers=[  # 包的分类信息
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # 使用 MIT License
        "Operating System :: OS Independent",
    ],
)
