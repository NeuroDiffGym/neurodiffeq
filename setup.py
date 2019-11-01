import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neurodiffeq",
    version="1.0.0",
    author="odegym",
    author_email="feiyu_chen@g.harvard.edu",
    description="A Python package for solving differential equations with neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/odegym/neurodiffeq",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)