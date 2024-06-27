import os

from fastapi import UploadFile


async def save_uploaded_file(uploaded_file: UploadFile, save_dest: str) -> str:
    # TODO: replace this with s3 and use save_dest
    file_path = os.path.join(save_dest, uploaded_file.filename)
    with open(file_path, 'wb') as file:
        contents = await uploaded_file.read()
        file.write(contents)
    return file_path


def make_dir(fp: str):
    if not os.path.exists(fp):
        os.mkdir(fp)


