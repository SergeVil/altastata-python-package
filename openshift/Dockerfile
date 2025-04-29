FROM quay.io/opendatahub-contrib/workbench-images:jupyter-datascience-c9s-py311_2023c_latest

USER root

# Install Java 17
RUN dnf install -y java-17-openjdk-devel && \
    dnf clean all

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk

# Create all necessary directories with proper permissions
RUN mkdir -p /opt/app-root/src/.jupyter && \
    mkdir -p /opt/app-root/src/.local/share/jupyter/metadata/runtimes && \
    mkdir -p /home/jovyan/.jupyter && \
    mkdir -p /opt/app-root/lib64/python3.11/site-packages/parso/python && \
    touch /opt/app-root/lib64/python3.11/site-packages/parso/python/grammar311.txt && \
    chown -R 1001:0 /opt/app-root/src /home/jovyan && \
    chmod -R g+rwX /opt/app-root/src /home/jovyan

# Copy requirements files
COPY --chown=1001:0 openshift/requirements.txt ./
COPY --chown=1001:0 openshift/pytorch-cpu.txt ./

# Install regular dependencies first
RUN echo "Installing regular packages" && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt

# Install PyTorch CPU-only versions
RUN echo "Installing PyTorch CPU packages" && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed -r pytorch-cpu.txt --index-url https://download.pytorch.org/whl/cpu

# Create package directory and copy files
RUN mkdir -p /opt/app-root/src/altastata-package
COPY --chown=1001:0 setup.py README.md /opt/app-root/src/altastata-package/
COPY --chown=1001:0 altastata /opt/app-root/src/altastata-package/altastata/
# Copy PyTorch examples
COPY --chown=1001:0 pytorch-example /opt/app-root/src/pytorch-example/

# Set proper permissions for package installation
RUN chown -R 1001:0 /opt/app-root/src/altastata-package /opt/app-root/src/pytorch-example && \
    chmod -R g+rwX /opt/app-root/src/altastata-package /opt/app-root/src/pytorch-example

# Switch to non-root user for remaining operations
USER 1001

WORKDIR /opt/app-root/src/altastata-package

# Install the package in development mode and verify installation
RUN pip install -e . && \
    echo "Directory structure:" && \
    ls -la /opt/app-root/src/altastata-package && \
    echo "\nPackage directory:" && \
    ls -la /opt/app-root/src/altastata-package/altastata && \
    echo "\nPython path:" && \
    python -c "import sys; print('\n'.join(sys.path))" && \
    echo "\nTrying to import:" && \
    python -c "import altastata; print('Module location:', altastata.__file__)" && \
    python -c "from altastata import AltaStataFunctions; print('AltaStataFunctions imported successfully')"

# Configure Jupyter
WORKDIR /home/jovyan
RUN echo '{"ServerApp": {"token": "", "allow_origin": "*", "allow_credentials": true}}' > /home/jovyan/.jupyter/jupyter_server_config.json

# Expose Jupyter port
EXPOSE 8888

# Set environment variables for Jupyter
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=""
ENV JUPYTER_CONFIG_DIR=/home/jovyan/.jupyter

# Set the image label
LABEL org.opencontainers.image.title="altastata/jupyter-datascience"
LABEL org.opencontainers.image.description="Jupyter DataScience environment with AltaStata integration"
LABEL org.opencontainers.image.version="latest"

CMD ["start-notebook.sh"]
