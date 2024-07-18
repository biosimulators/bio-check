# BioCheck: A Simulation Verification Service API
#### __This service utilizes separate containers for API management, job processing, and datastorage with MongoDB, ensuring scalable and robust performance.__

## Getting Started:

### **HIGH-LEVEL `bio_check` API:**
The primary method of user-facing interaction for this service is done through the use of a high-level "notebook" api called `bio_check`. Installation of this tooling
can be performed using PyPI as such:

`pip install bio-check`

**PLEASE NOTE: You must have `>=python3.9` in order to use the high-level api.**

### **GOOGLE COLAB DEMO:**
A convenient notebook demonstrating the functionality of this service is hosted on Google Colab and can be found [here.](https://colab.research.google.com/drive/19uxh93pZvhCGXkC7a15SmAx4oH4MV7OJ#scrollTo=j_mN-EE3vanZ)
### **FOR DEVELOPERS:**

This application (`bio_check`) uses a microservices architecture which presents the following libraries:

- `api`: This library handles all requests including saving uploaded files, pending job creation, fetching results, and contains the user-facing endpoints.
- `storage`: This library handles MongoDB configs as well as bucket-like storages for uploaded files.
- `worker`: This library handles all job processing tasks for verification services such as job status adjustment, job retrieval, and comparison execution.

The installation process is outlined as follows:

1. `git clone https://github.com/biosimulators/bio-check.git`
2. `cd bio-check/bio_check`
3. `touch .env`
4. Enter the following fields into the `.env` file: `MONGO_DB_USERNAME, MONGO_DB_PWD, MONGO_DB_URI`.
5. **Ensure that your IP address has been authorized in the `bio-check` cluster in Mongo Atlas.**


### Notes:
- This application currently uses MongoDB as the database store in which jobs are read/written. Database access is given to both the `api` and `worker` libraries. Such database access is 
executed/implemented with the use of a `Supervisor` singleton. **TODO: Make this pattern implementation threadsafe.**


### **Note (05/22/2024):**
The only package source that is currently supported by this tooling is `PyPI`. The support of other potential 
package sources such as `conda`, `brew`, `apt`, and more is currently under development.
