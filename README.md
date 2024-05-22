### Simulation Verification Service API.

- The `src` library serves as the "source" content, for which the REST API serves to implement
and enforce.

- The `server` library serves to define a FastAPI server.


When the functions from `server` are called, instances of the objects from `src` are created. 
It is then up to the client to implement these rest calls correctly.
