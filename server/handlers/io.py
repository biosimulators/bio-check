import tempfile

from biosimulators_utils.combine.io import CombineArchiveReader


def unpack_omex(archive_fp: str, save_dir: str):
    return CombineArchiveReader().run(archive_fp, save_dir)


def save_omex_archive(contents: bytes, save_dir: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.omex') as temp_file:
        temp_file.write(contents)
        archive_path = temp_file.name

    return {'source': archive_path, 'archive': unpack_omex(archive_path, save_dir), 'save_dir': save_dir}

