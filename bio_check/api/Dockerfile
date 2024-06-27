 # Api container

FROM ghcr.io/biosimulators/bio-check-base:latest

EXPOSE 3001

# for API
# tensorstore = "^0.1.63"
# aiofiles = "^24.1.0"
# aiobotocore = "^2.13.1"
# pyyaml = "^6.0.1"
# httpx = "^0.27.0"

COPY requirements.api.txt ./assets/requirements.api.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r ./assets/requirements.api.txt

CMD ["uvicorn", "bio_check.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]
