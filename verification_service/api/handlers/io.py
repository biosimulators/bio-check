from fastapi import UploadFile


# TODO: Configure this with an S3-like and import.
async def save_uploaded_file(file: UploadFile):
    out_file_path = f"/path/to/shared/storage/{file.filename}"
    with open(out_file_path, 'wb') as out_file:
        content = await file.read()
        out_file.write(content)
    return out_file_path
