import re
import setuptools

with open("README.md", "r") as fh:
    def func(m):
        match = m.group()
        url = m.group(1)
        if url.startswith('resources/'):
            return match.replace(url, 'https://raw.githubusercontent.com/NeuroDiffGym/neurodiffeq/master/' + url)
        return match

    long_description = fh.read()
    long_description = re.sub(r"!\[.*?\]\((.*?)\)", func, long_description)

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh.read().split('\n') if line.strip() != '']

setuptools.setup(
    name="neurodiffeq",
    version="0.3.5",
    author="neurodiffgym",
    author_email="shuheng_liu@g.harvard.edu",
    description="A light-weight & flexible library for solving differential equations using neural networks based on PyTorch. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeuroDiffGym/neurodiffeq",
    download_url="https://github.com/NeuroDiffGym/neurodiffeq/archive/v0.3.5.tar.gz",
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
    install_requires=requirements,
)
