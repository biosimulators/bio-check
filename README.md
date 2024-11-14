![GitHub CI](https://github.com/biosimulators/bio-check/actions/workflows/ci.yml/badge.svg)
![Base Deployment](https://github.com/biosimulators/bio-check/actions/workflows/deploy-base.yml/badge.svg)
![Microservices Deployment](https://github.com/biosimulators/bio-check/actions/workflows/deploy-microservices.yml/badge.svg)
# BioCheck (bio-compose-server): A Biological Simulation Verification Service
### __This service utilizes separate containers for REST API management, job processing, and datastorage with MongoDB, ensuring scalable and robust performance.__

## **The REST API can be accessed via Swagger UI here: [https://biochecknet.biosimulations.org/docs](https://biochecknet.biosimulations.org/docs)

## **For Developers:**

### This application ("BioCompose") uses a microservices architecture which presents the following libraries:

- `api`: This library handles all requests including saving uploaded files, pending job creation, fetching results, and contains the user-facing endpoints.
- `storage`: This library handles MongoDB configs as well as bucket-like storages for uploaded files.
- `worker`: This library handles all job processing tasks for verification services such as job status adjustment, job retrieval, and comparison execution.

### The simulators used by this application consist of multiple python language bindings of C/C++ libraries. Given this fact, is it helpful to be aware of the dependency network required by each simulator. See the following documentation for simulators used in this application:

- [AMICI](https://amici.readthedocs.io/en/latest/python_installation.html)
- [COPASI(basico)](https://basico.readthedocs.io/en/latest/quickstart/get-started.html#installation)
- [PySCes](https://pyscesdocs.readthedocs.io/en/latest/userguide_doc.html#installing-and-configuring)
- [Tellurium](https://tellurium.readthedocs.io/en/latest/installation.html)
- [Simulator-specific implementations of the Biosimulators-Utils interface](https://docs.biosimulations.org/users/biosimulators-packages)
- [Smoldyn](https://www.smoldyn.org/SmoldynManual.pdf)
- *(Coming soon:)* [ReaDDy](https://readdy.github.io/installation.html)


### Dependency management scopes are handled as follows:

#### _*Locally/Dev*_:
- Anaconda via `environment.yml` - the closest to local development at root level which mimics what actually happens in the containers (conda deps tend to break more frequently than poetry.)

_*Remotely in microservice containers*_:
- Remote microservice container management is handled by `conda` via `environment.yml` files for the respective containers.

### The installation process is outlined as follows:

1. `git clone https://github.com/biosimulators/bio-check.git`
2. `cd assets`
3. `touch .env_dev`
4. Enter the following fields into the `.env_dev` file: 
        
        MONGO_URI=<uri of your mongo instance. In this case we use the standard mongodb image with the app name bio-check>
        GOOGLE_APPLICATION_CREDENTIALS=<path to your gcloud credentials .json file. Contact us for access>
        BUCKET_NAME=bio-check-requests-1  # name of the bucket used in this app
5. `cd ..`
6. Pull and run the latest version of Mongo from the Docker Hub. (`docker run -d -it mongo:latest` or similar.)
7. Create a conda env from the environment file at the root of this repo:
         
        conda env create -f environment.yml -y && conda activate bio-composer-server-dev
8. Install pysces with conda and amici with pip:
   
        conda install -c conda-forge -c pysces pysces
        conda run pip3 install biosimulators-amici  # installs both biosimulators and amici
9. If using Smoldyn, there is a arm-based mac installation script in `assets/dev/` called `install-smoldyn-mac-silicon.sh`. So run the following:

        sudo chmod +x ./assets/dev/scripts/install-smoldyn-mac-silicon.sh  # or whichever method you are using to install
        ./assets/dev/scripts/install-smoldyn-mac-silicon.sh  # conda is configured to install Smoldyn into its environment


## Notes:
- This application currently uses MongoDB as the database store in which jobs are read/written. Database access is given to both the `api` and `worker` libraries. Such database access is 
executed/implemented with the use of a `Supervisor` singleton.

