import tempfile
from typing import *

import uvicorn
from pydantic import Field
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware

from server.handlers.io import save_omex_archive
from server.handlers.compare import generate_utc_species_comparison
from server.data_model import ArchiveUploadResponse, UtcSpeciesComparison


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


# @app.post(
#     "/upload-omex",
#     response_model=ArchiveUploadResponse,
#     summary='Upload an OMEX archive')
# async def upload_omex(file: UploadFile = File(...), save_location: str = Field(...)) -> ArchiveUploadResponse:
#     contents = await file.read()
#     save_dir = save_location or tempfile.mkdtemp()
#     archive_response = save_omex_archive(contents, save_dir)
#     return ArchiveUploadResponse(filename=archive_response['source'])


@app.post(
    "/utc-species-comparison",
    response_model=UtcSpeciesComparison,
    summary="Compare UTC outputs for a given species name")
async def utc_species_comparison(
        species_id: str,
        save_location: str = None,
        omex_fp: str = None,
        file: UploadFile = File(...),
        simulators: List[str] = Body(default=['amici', 'copasi', 'tellurium']),
) -> UtcSpeciesComparison:
    contents = await file.read()
    save_dir = save_location or tempfile.mkdtemp()
    archive_response = save_omex_archive(contents, save_dir)
    comparison = generate_utc_species_comparison(
        omex_fp=omex_fp,
        out_dir=save_dir,
        species_name=species_id,
        simulators=simulators)

    print(comparison)
    return UtcSpeciesComparison(mse=comparison['mse'], proximity=comparison['prox'])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
