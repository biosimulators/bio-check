## Simulation Verification Service API.

This application (`verification_service`) uses a microservices architecture which presents the following libraries:

- `api`: This library handles all requests including saving uploaded files, pending job creation, fetching results, and contains the user-facing endpoints.
- `storage`: This library handles MongoDB configs as well as bucket-like storages for uploaded files.
- `worker`: This library handles all job processing tasks for verification services such as job status adjustment, job retrieval, and comparison execution.


### Getting Started:

#### __For Developers__:
1. `git clone https://github.com/biosimulators/verification-service.git`
2. `cd verification-service/verification_service`
3. `touch .env`
4. Enter the following fields into the `.env` file: `MONGO_DB_USERNAME, MONGO_DB_PWD, MONGO_DB_URI`.
5. **Ensure that your IP address has been authorized in the `verification-service` cluster in Mongo Atlas.**


### Notes:
- This application currently uses MongoDB as the database store in which jobs are read/written. Database access is given to both the `api` and `worker` libraries. Such database access is 
executed/implemented with the use of a `Supervisor` singleton. **TODO: Make this pattern implementation threadsafe.**


### **Note (05/22/2024):**
The only package source that is currently supported by this tooling is `PyPI`. The support of other potential 
package sources such as `conda`, `brew`, `apt`, and more is currently under development.
