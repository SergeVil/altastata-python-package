from setuptools import setup, find_packages

setup(
    name='altastata',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'altastata_api': ['lib/*.jar']
    },
    install_requires=[
        'py4j==0.10.9.5',
    ],
)

