import os
import tempfile
import shutil
from typing import *
from zipfile import ZipFile

import uvicorn
from pydantic import Field
from fastapi import FastAPI, UploadFile, File, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from server.handlers.io import save_omex_archive, unpack_omex
from server.handlers.compare import generate_biosimulators_utc_species_comparison, generate_utc_comparison
from server.data_model import ArchiveUploadResponse, UtcSpeciesComparison, UtcComparison


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


@app.get("/")
def root():
    return {'verification-service-message': 'Hello from the Verification Service API!'}


@app.post(
    "/utc-comparison",
    response_model=UtcComparison,
    summary="Compare UTC outputs for each species in a given model file")
async def utc_comparison(
        uploaded_file: UploadFile = File(...),
        simulators: List[str] = Query(default=['amici', 'copasi', 'tellurium']),
        include_outputs: bool = Query(default=True),
        comparison_id: str = Query(default=None)
) -> UtcComparison:
    # handle os structures
    save_dir = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()
    omex_path = os.path.join(save_dir, uploaded_file.filename)
    with open(omex_path, 'wb') as file:
        contents = await uploaded_file.read()
        file.write(contents)

    comparison_name = comparison_id or f'api-generated-utc-comparison-for-{simulators}'
    # generate async comparison
    comparison = await generate_utc_comparison(
        omex_fp=omex_path,
        simulators=simulators,
        include_outputs=include_outputs,
        comparison_id=comparison_name)

    spec_comparisons = []
    for spec_name, comparison_data in comparison['results'].items():
        species_comparison = UtcSpeciesComparison(
            mse=comparison_data['mse'],
            proximity=comparison_data['prox'],
            output_data=comparison_data.get('output_data'))
        spec_comparisons.append(species_comparison)

    return UtcComparison(results=spec_comparisons, id=comparison_name, simulators=simulators)


@app.post(
    "/biosimulators-utc-species-comparison",
    response_model=UtcSpeciesComparison,
    summary="Compare UTC outputs from Biosimulators for a given species name")
async def biosimulators_utc_species_comparison(
        uploaded_file: UploadFile = File(...),
        species_id: str = Query(...),
        simulators: List[str] = Query(default=['amici', 'copasi', 'tellurium']),
        include_outputs: bool = Query(default=True)
) -> UtcSpeciesComparison:
    # handle os structures
    save_dir = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()
    omex_path = os.path.join(save_dir, uploaded_file.filename)
    with open(omex_path, 'wb') as file:
        contents = await uploaded_file.read()
        file.write(contents)

    # generate async comparison
    comparison = await generate_biosimulators_utc_species_comparison(
        omex_fp=omex_path,
        out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
        species_name=species_id,
        simulators=simulators)

    out_data = comparison['output_data'] if include_outputs else None

    return UtcSpeciesComparison(
        mse=comparison['mse'],
        proximity=comparison['prox'],
        output_data=out_data)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
