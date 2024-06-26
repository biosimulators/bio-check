 # Api container

FROM ghcr.io/biosimulators/bio-check-base:latest

EXPOSE 3001

# libSBMLSIM
# ARG LSBMLSIM_VERSION=1.4.0
# ARG LIBSBML_VERSION=5.19.0
#
# # libSBML
# RUN apt-get install --no-install-recommends -y \
#         wget \
#         libxml2 \
#     \
#     && wget https://master.dl.sourceforge.net/project/sbml/libsbml/${LIBSBML_VERSION}/stable/Linux/64-bit/libSBML-${LIBSBML_VERSION}-Linux-x64.deb \
#     && dpkg -i libSBML-${LIBSBML_VERSION}-Linux-x64.deb \
#     \
#     && rm libSBML-${LIBSBML_VERSION}-Linux-x64.deb
# ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
#
# # LibSBMLSim
# RUN apt-get install --no-install-recommends -y \
#         wget \
#         build-essential \
#         cmake \
#         swig \
#         libbz2-dev \
#     \
#     && PY_EXEC_PATH=$(which python) \
#     && PY_PREFIX=$(echo ${PY_EXEC_PATH} | rev | cut -d "/" -f 3- | rev) \
#     && cd /tmp \
#     && wget https://github.com/libsbmlsim/libsbmlsim/archive/refs/tags/v${LSBMLSIM_VERSION}.tar.gz \
#     && tar xvvf v${LSBMLSIM_VERSION}.tar.gz \
#     && cd libsbmlsim-${LSBMLSIM_VERSION} \
#     && mkdir build \
#     && cd build \
#     && PYTHON_MAJOR_MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d . -f 1-2) \
#     && cmake \
#         -D LIBSBML_INCLUDE_DIR=/usr/include \
#         -D LIBSBML_LIBRARY=/usr/lib64/libsbml.so \
#         -D WITH_PYTHON=ON \
#         -D PYTHON_PREFIX=${PY_PREFIX} \
#         -D PYTHON_EXECUTABLE:FILEPATH=${PY_EXEC_PATH} \
#         -D PYTHON_INCLUDE_DIR:PATH=/usr/local/include/python${PYTHON_MAJOR_MINOR_VERSION} \
#         -D PYTHON_LIBRARY:FILEPATH=/usr/local/lib/libpython${PYTHON_MAJOR_MINOR_VERSION}.so \
#         .. \
#     \
#     && make \
#     && make install \
#     \
#     && cd /tmp \
#     && rm v${LSBMLSIM_VERSION}.tar.gz \
#     && rm -r libsbmlsim-${LSBMLSIM_VERSION}

COPY requirements.api.txt ./assets/requirements.api.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r ./assets/requirements.api.txt

CMD ["uvicorn", "bio_check.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]
