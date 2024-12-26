![Deploy API](https://github.com/biosimulators/bio-check/actions/workflows/deploy-gateway.yml/badge.svg)
![Deploy Worker](https://github.com/biosimulators/bio-check/actions/workflows/deploy-worker.yml/badge.svg)

# BioCompose Server: A Biological Simulation Verification Service
### __This service utilizes separate containers for REST API management, job processing, and datastorage with MongoDB, ensuring scalable and robust performance.__

## **The REST API can be accessed via Swagger UI here: [https://compose.biosimulations.org/docs](https://compose.biosimulations.org/docs)

## **For Developers:**

### This application ("BioCompose") uses a microservices architecture which presents the following libraries:

- `gateway`: This library handles all requests including saving uploaded files, pending job creation, fetching results, and contains the user-facing endpoints.
- `shared`: A library of common objects/pointers used by both `gateway` and `worker`.
- `worker`: This library handles all job processing tasks for verification services such as job status adjustment, job retrieval, and more.

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

1. `git clone https://github.com/biosimulators/bio-compose-server.git`
2. `cd bio-compose-server/shared`
3. `mv .env_template .env`
4. Enter the following fields into the `.env` file: 
        
        MONGO_URI=<uri of your mongo instance. In this case we use the standard mongodb image with the app name bio-check>
        GOOGLE_APPLICATION_CREDENTIALS=<path to your gcloud credentials .json file. Contact us for access>
        BUCKET_NAME=bio-check-requests-1  # name of the bucket used in this app
5. `cd ..`
6. Pull and run the latest version of Mongo from the Docker Hub. (`docker run -d -it mongo:latest` or similar.)
7. `sudo chmod +x ./assets/dev/scripts/install.sh`
8. `./install.sh`


## Notes:
- This application currently uses MongoDB as the database store in which jobs are read/written. Database access is given to both the `api` and `worker` libraries. Such database access is 
executed/implemented with the use of a `Supervisor` singleton.

