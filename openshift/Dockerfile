FROM --platform=linux/amd64 python:3.10-slim

# System setup
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Directory structure
WORKDIR /opt/app-root
RUN mkdir -p /opt/app-root/src/altastata-package

# Python dependencies
COPY openshift/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Package installation
COPY setup.py README.md /opt/app-root/src/altastata-package/
COPY altastata /opt/app-root/src/altastata-package/altastata/
WORKDIR /opt/app-root/src/altastata-package
RUN pip install -e .

# Jupyter configuration
ENV JUPYTER_TOKEN="" \
    JUPYTER_CONFIG_DIR=/opt/app-root/.jupyter \
    PYTHONPATH=/opt/app-root/src

RUN mkdir -p /opt/app-root/.jupyter && \
    jupyter lab --generate-config && \
    echo "c.ServerApp.token = ''" >> /opt/app-root/.jupyter/jupyter_lab_config.py

# Verification
RUN python -c "from altastata import AltaStataFunctions; print('Package verification successful')"

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 