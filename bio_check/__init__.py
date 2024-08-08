import os

from bio_check.verifier import Verifier


current_dir = os.path.dirname(__file__)
version_file_path = os.path.join(current_dir, '_VERSION')

with open(version_file_path, 'r') as f:
    __version__ = f.read().strip()

