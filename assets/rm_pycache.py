import os
import sys
import shutil


libs = ['compose_api', 'compose_worker']


for lib in libs:
    files = os.listdir(f'./{lib}')
    for f in files:
        if f.startswith('__pycache'):
            shutil.rmtree(os.path.join('.', lib, f))

