from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='altastata',
    version='0.0.6',
    author='Serge Vilvovsky',
    author_email='serge.vilvovsky@altastata.com',
    description='A Python package for Altastata data processing and machine learning integration',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/sergevil/altastata-python-package',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # Bundle the unified mycloud runtime uber jar plus signed Bouncy Castle
        # jars, and include the packaged Console UI static bundle.
        'altastata': [
            'lib/altastata-services-*-uber.jar',
            'lib/bc*.jar',
            'grpc/v1/*.py',
            'lib/altastata-console-static/*',
            'lib/altastata-console-static/*/*',
        ],
        '': ['proto/**/*.proto']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'fsspec>=2023.1.0',
        'grpcio>=1.69.0',
        'protobuf>=4.28.3',
    ],
    entry_points={
        'console_scripts': [
            'altastata-grpc-server=altastata.grpc_server:main',
        ],
    },
)

