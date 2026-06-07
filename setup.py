from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='altastata',
    version='0.1.38',
    author='Serge Vilvovsky',
    author_email='serge.vilvovsky@altastata.com',
    description='A Python package for Altastata data processing and machine learning integration',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/sergevil/altastata-python-package',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # The lib/* glob captures the bundled altastata-grpc uber jar plus
        # py4j and Bouncy Castle runtime jars. The two altastata-console-static
        # patterns walk the SPA bundle one and two levels deep (Vite emits
        # only an `assets/` subdir, so two levels is enough today; the second
        # glob future-proofs against deeper nesting). All these binary
        # artifacts are gitignored under altastata/lib/ and are populated
        # locally by scripts/build-bundled-artifacts.sh before
        # `python -m build`.
        'altastata': [
            'lib/*.jar',
            'v1/*.py',
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
        'py4j==0.10.9.5',
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

