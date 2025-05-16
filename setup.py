from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="XpertEval",
    version="0.1.0",
    author="rookielittleblack",
    author_email="rookielittleblack@yeah.net",
    description="全模态大模型一站式评测框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rookie-littleblack/XpertEval",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True
)
