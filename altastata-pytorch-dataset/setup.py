from setuptools import setup, find_packages

setup(
    name="altastata-pytorch-dataset",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "Pillow"
    ],
    extras_require={
        "cloud": ["altastata-python-package"]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A custom PyTorch Dataset for reading files from both local storage and AltaStata cloud storage",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/altastata-pytorch-dataset",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 