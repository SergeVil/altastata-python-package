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

