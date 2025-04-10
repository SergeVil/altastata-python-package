FROM --platform=linux/amd64 quay.io/opendatahub-contrib/workbench-images:jupyter-datascience-c9s-py311_2023c_latest

# System setup
USER root
RUN dnf install -y java-17-openjdk-devel && \
    dnf clean all

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Directory structure
WORKDIR /opt/app-root
RUN mkdir -p /opt/app-root/src/altastata-package && \
    mkdir -p /opt/app-root/src/.jupyter && \
    mkdir -p /opt/app-root/src/.local/share/jupyter && \
    chown -R 1001:0 /opt/app-root && \
    chmod -R g+rwX /opt/app-root

# Python dependencies
COPY --chown=1001:0 openshift/requirements.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/requirements.txt && \
    mkdir -p /opt/app-root/lib64/python3.11/site-packages/parso/python && \
    touch /opt/app-root/lib64/python3.11/site-packages/parso/python/grammar311.txt

# Package installation
COPY --chown=1001:0 setup.py README.md /opt/app-root/src/altastata-package/
COPY --chown=1001:0 altastata /opt/app-root/src/altastata-package/altastata/
WORKDIR /opt/app-root/src/altastata-package
RUN pip install -e .

# Jupyter configuration
ENV JUPYTER_TOKEN="" \
    JUPYTER_CONFIG_DIR=/opt/app-root/.jupyter \
    JUPYTER_DATA_DIR=/opt/app-root/src/.local/share/jupyter \
    JUPYTER_RUNTIME_DIR=/opt/app-root/src/.local/share/jupyter/runtime \
    PYTHONPATH=/opt/app-root/src

# Configure Jupyter and install kernel
RUN mkdir -p /opt/app-root/.jupyter && \
    jupyter lab --generate-config && \
    echo "c.ServerApp.token = ''" >> /opt/app-root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /opt/app-root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /opt/app-root/.jupyter/jupyter_lab_config.py && \
    python -m ipykernel install --user --name=python3 --display-name="Python 3" && \
    jupyter lab clean && \
    chown -R 1001:0 /opt/app-root/.jupyter && \
    chmod -R g+rwX /opt/app-root/.jupyter

# Switch back to non-root user
USER 1001

# Verification
RUN python -c "from altastata import AltaStataFunctions; print('Package verification successful')"

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 