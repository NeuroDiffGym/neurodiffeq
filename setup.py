import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neurodiffeq",
    version="0.1.1",
    author="odegym",
    author_email="wish1104@icloud.com",
    description="A light-weight & flexible library for solving differential equations using neural networks based on PyTorch. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/odegym/neurodiffeq",
    download_url="https://github.com/odegym/neurodiffeq/archive/v0.1.1.tar.gz",
    keywords=[
        "neural network", 
        "deep learning", 
        "differential equation", 
        "ODE", 
        "PDE", 
        "spectral method",
        "numerical method",
        "pytorch", 
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
