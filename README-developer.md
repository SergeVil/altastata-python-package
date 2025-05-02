# Make sure you have py4j0.10.9.8.jar or similar at altastata/lib directory

# for example for Windows
cp /c/Users/serge/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/share/py4j/py4j0.10.9.8.jar altastata/lib/

# Make sure you have altastata-hadoop jar (created without bouncy castle) and separate bouncy castle jars

# for example
# go to altastata-hadoop

gradle clean build shadowJar -PexcludeBouncyCastle=true copyDeps

# to build this one

cp ../mycloud/altastata-hadoop/build/libs/altastata-hadoop-all.jar altastata/lib/
cp ../mycloud/altastata-hadoop/build/libs_dependency/bc*-jdk18on-*.jar altastata/lib/

# verify that the jar is ok (it was corrupted in Linux)
jar tf altastata/lib/py4j0.10.9.5.jar | grep GatewayServer

# if py4j file is corrupted, run
wget https://repo1.maven.org/maven2/net/sf/py4j/py4j/0.10.9.5/py4j-0.10.9.5.jar -O altastata/lib/py4j0.10.9.5.jar

# if you want to change the logs level copy and modify this file
cp ../mycloud/altastata-hadoop/src/main/resources/logback.xml altastata/lib/

# install
pip install -e .

# test
python test_script.py

# build docker
docker buildx build --platform linux/amd64,linux/arm64 --push -t ghcr.io/sergevil/altastata/jupyter-datascience:2024a_latest -f openshift/Dockerfile .

# push to the registry if needed
docker push ghcr.io/sergevil/altastata/jupyter-datascience:2024a_latest

# run docker
docker run --name altastata-jupyter -d -p 8888:8888 -v /Users/sergevilvovsky/.altastata:/opt/app-root/src/.altastata:rw -v /Users/sergevilvovsky/Desktop:/opt/app-root/src/Desktop:rw ghcr.io/sergevil/altastata/jupyter-datascience:2024a_latest

# Building and Uploading to PyPI

# 1. Install required tools
pip install --upgrade pip
pip install --upgrade build
pip install --upgrade twine

# 2. Clean previous builds
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/

# 3. Build the package
python -m build

# 4. Verify the built package
twine check dist/*

# 5. Upload to PyPI Test (optional, for testing)
# twine upload --repository testpypi dist/*

# 6. Upload to PyPI Production
twine upload dist/*

# Note: You'll need to have a PyPI account and API token
# To create an API token:
# 1. Go to https://pypi.org/manage/account/token/
# 2. Create a new token with appropriate permissions
# 3. Store the token securely
# 4. Create or update ~/.pypirc with:
# [pypi]
# username = __token__
# password = pypi-your-token-here

# For large files (like JARs), you may need to request a file size limit increase from PyPI
# https://github.com/pypi/support/issues/6225
