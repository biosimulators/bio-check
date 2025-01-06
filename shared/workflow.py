"""
A.P:
The general workflow should be:

1. gateway: client uploads JSON spec file
2. gateway: gateway stores the JSON spec as a job in mongo
3. gateway: spec is returned to client as confirmation
4. worker: check for pending statuses in mongo collection (single)
5. worker: get simulators/deps from #4 TODO: somehow do this!!
6. worker: dynamic install #5
7. worker: change job status in DB to IN_PROGRESS
8. worker: run composition with pbg.Composite() and gather_results()
9. worker: change job status in DB to COMPLETE
10. worker: update job document ['results'] field with #8's data
11. worker: perhaps emit an event?
"""