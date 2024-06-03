import tempfile

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from server.handlers.io import save_omex_archive
from server.handlers.compare import generate_utc_species_comparison
from server.data_model import ArchiveUploadResponse


app = FastAPI(title='verification-service')


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


@app.post(
    "/upload-omex",
    response_model=ArchiveUploadResponse,
    summary='Upload an OMEX archive')
async def upload_omex(file: UploadFile = File(...)):
    contents = await file.read()
    save_dir = tempfile.mkdtemp()
    archive_response = download_file(contents, save_dir)
    return ArchiveUploadResponse(filename=archive_response['source'])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
