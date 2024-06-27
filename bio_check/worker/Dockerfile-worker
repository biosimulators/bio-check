FROM ghcr.io/biosimulators/bio-check-base:latest

RUN useradd -r -g appgroup worker

USER worker

RUN poetry add \
    biosimulators-copasi \
    biosimulators-amici \
    biosimulators-tellurium \
    biosimulators-utils --extras=logging \
    biosimulator-processes
    # h5py

COPY bio_check/worker .

# TODO: For CMD, run a main python file that is a while loop checking for jobs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
