import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from biosimulator_processes.server.src.io import save_omex_archive


app = FastAPI(title='biosimulator-processes-server')


origins = [
    "http://localhost:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-omex")
async def upload_omex(file: UploadFile = File(...)):
    contents = await file.read()

    save_dir = tempfile.mkdtemp()
    archive_response = save_omex_archive(contents, save_dir)
    print(archive_response['archive'].contents)
    return {"filename": archive_response['source']}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
