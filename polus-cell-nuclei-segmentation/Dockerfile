FROM labshare/polus-bfio-util:2.0.6-tensorflow

COPY VERSION /

ARG EXEC_DIR="/opt/executables"

#Create folders
RUN mkdir -p ${EXEC_DIR}

#Copy executable
COPY src ${EXEC_DIR}/

RUN pip install --upgrade setuptools && \
    pip install -r ${EXEC_DIR}/requirements.txt --no-cache-dir && \
    mv ${EXEC_DIR}/tensorflow_backend.py /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py -f 

RUN cd /opt/executables && \
    sh get_models.sh && \
    python get_resnet_weights.py

ENTRYPOINT [ "python3", "models.py" ]
