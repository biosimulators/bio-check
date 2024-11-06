![GitHub CI](https://github.com/biosimulators/bio-check/actions/workflows/ci.yaml/badge.svg)
![GitHub CD](https://github.com/biosimulators/bio-check/actions/workflows/cd.yaml/badge.svg)
# BioCheck (bio-compose-server): A Biological Simulation Verification Service
#### __This service utilizes separate containers for REST API management, job processing, and datastorage with MongoDB, ensuring scalable and robust performance.__

## **The REST API can be accessed via Swagger UI here: [https://biochecknet.biosimulations.org/docs](https://biochecknet.biosimulations.org/docs)

## **FOR DEVELOPERS:**

This application (`bio_check`) uses a microservices architecture which presents the following libraries:

- `api`: This library handles all requests including saving uploaded files, pending job creation, fetching results, and contains the user-facing endpoints.
- `storage`: This library handles MongoDB configs as well as bucket-like storages for uploaded files.
- `worker`: This library handles all job processing tasks for verification services such as job status adjustment, job retrieval, and comparison execution.

Dependency management scopes are handled as follows:

_*Locally/Dev*_:
- Python poetry via `pyproject.toml` - the most stable/reproducible method, yet the farthest from what is actually happening in the service containers as they use conda.
- Anaconda via `environment.yml` - the closest to local development at root level which micics what actually happens in the containers (conda deps tend to break more frequently than poetry.)

_*Remotely in microservice containers*_:
- Remote microservice container management is handled by `conda` via `environment.yml` files for the respective containers.

The installation process is outlined as follows:

1. `git clone https://github.com/biosimulators/bio-check.git`
2. `cd assets`
3. `touch .env_dev`
4. Enter the following fields into the `.env_dev` file: 
        
        MONGO_URI=<uri of your mongo instance. In this case we use the standard mongodb image with the app name bio-check>
        GOOGLE_APPLICATION_CREDENTIALS=<path to your gcloud credentials .json file. Contact us for access>
        BUCKET_NAME=bio-check-requests-1  # name of the bucket used in this app
5. `cd ..`
6. Pull and run the latest version of Mongo from the Docker Hub. (`docker run -d -it mongo:latest` or similar.)
7. Run one of the following commands based on your preference:
         
        conda env create -n bio-compose-server-dev -f environment.yml
        
        # OR
        poetry env use 3.10 && poetry install


### Notes:
- This application currently uses MongoDB as the database store in which jobs are read/written. Database access is given to both the `api` and `worker` libraries. Such database access is 
executed/implemented with the use of a `Supervisor` singleton.

