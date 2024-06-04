import os
import tempfile
import shutil
from typing import *
from zipfile import ZipFile

import uvicorn
from pydantic import Field
from fastapi import FastAPI, UploadFile, File, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from server.handlers.io import save_omex_archive, unpack_omex
from server.handlers.compare import generate_utc_species_comparison
from server.data_model import ArchiveUploadResponse, UtcSpeciesComparison


app = FastAPI(title='verification-service')
router = APIRouter()

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


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post('/upload-OMEX-archive')
async def upload_archive(uploaded_file: UploadFile = File(...)):
    save_dir = "/tmp/stuff"
    content = await uploaded_file.read()
    with open(save_dir, "wb") as tozip:
        tozip.write(content)
    with ZipFile(uploaded_file.filename, 'w') as myzip:
        myzip.write(save_dir)
    response = FileResponse(path=uploaded_file.filename, filename=uploaded_file.filename)
    return response


@app.post("/exec-files")
async def create_file(uploaded_file: UploadFile = File(...)):
    contents = await uploaded_file.read()
    print(os.getcwd())
    save_dir = tempfile.mkdtemp()
    extraction_dir = tempfile.mkdtemp()

    omex_path = os.path.join(save_dir, uploaded_file.filename)
    with open(omex_path, 'w+b') as file:
        shutil.copyfileobj(uploaded_file.file, uploaded_file)

    with ZipFile(omex_path, 'w') as myzip:
        myzip.write(extraction_dir)

    print(os.path.exists(omex_path))
    print(os.listdir('.'))



@app.post(
    "/utc-species-comparison",
    response_model=UtcSpeciesComparison,
    summary="Compare UTC outputs for a given species name")
async def utc_species_comparison(
        species_id: str,
        uploaded_file: UploadFile = File(...),
        simulators: List[str] = Body(default=['amici', 'copasi', 'tellurium']),
) -> UtcSpeciesComparison:
    contents = await uploaded_file.read()
    print(os.getcwd())
    save_dir = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()

    omex_path = os.path.join(save_dir, uploaded_file.filename)
    with open(omex_path, 'w+b') as file:
        shutil.copyfileobj(uploaded_file.file, uploaded_file)

    print(os.path.exists(omex_path), os.path.exists(save_dir))
    comparison = generate_utc_species_comparison(
        omex_fp=omex_path,
        out_dir='tests/test_outputs',
        species_name=species_id,
        simulators=simulators)
    print(comparison)
    return UtcSpeciesComparison(
        mse=comparison['mse'],
        proximity=comparison['prox'],
        output_data=comparison['output_data'])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
