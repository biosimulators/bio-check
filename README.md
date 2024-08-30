![GitHub CI](https://github.com/biosimulators/bio-check/actions/workflows/ci.yaml/badge.svg)
![GitHub CD](https://github.com/biosimulators/bio-check/actions/workflows/cd.yaml/badge.svg)
# BioCheck (bio-compose-server): A Biological Simulation Verification Service
#### __This service utilizes separate containers for REST API management, job processing, and datastorage with MongoDB, ensuring scalable and robust performance.__

## Getting Started:

#### **The REST API can be accessed via Swagger UI here: [https://biochecknet.biosimulations.org/docs](https://biochecknet.biosimulations.org/docs)

### **FOR DEVELOPERS:**

This application (`bio_check`) uses a microservices architecture which presents the following libraries:

- `api`: This library handles all requests including saving uploaded files, pending job creation, fetching results, and contains the user-facing endpoints.
- `storage`: This library handles MongoDB configs as well as bucket-like storages for uploaded files.
- `worker`: This library handles all job processing tasks for verification services such as job status adjustment, job retrieval, and comparison execution.

The installation process is outlined as follows:

1. `git clone https://github.com/biosimulators/bio-check.git`
2. `cd assets`
3. `touch .env_dev`
4. Enter the following fields into the `.env_dev` file: 
        
        MONGO_URI=<uri of your mongo instance. In this case we use the standard mongodb image with the app name bio-check>
        GOOGLE_APPLICATION_CREDENTIALS=<path to your gcloud credentials .json file. Contact us for access>
        BUCKET_NAME=bio-check-requests-1  # name of the bucket used in this app

5. Pull the and run the latest version of Mongo from the Docker Hub. (`docker run -d -it mongo:latest` or similar.)


### Notes:
- This application currently uses MongoDB as the database store in which jobs are read/written. Database access is given to both the `compose_api` and `compose_worker` libraries. Such database access is 
executed/implemented with the use of a `Supervisor` singleton.

