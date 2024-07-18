# Create and publish a new version by creating and pushing a git tag for
# the version and publishing the version to PyPI. Also perform some
# basic checks to avoid mistakes in releases, for example tags not
# matching PyPI.


# set -e

version="$1"

if [ "${version}" == "" ]; then
  echo "You must enter a version to release as a runtime argument. Exiting."
  exit 1
fi

# version=$(grep "__version__" biosimulator_processes/_VERSION.py | awk -F\' '{print $2}')

# Check version is valid
setup_py_version="$(python setup.py --version)"
if [ "$setup_py_version" != "$version" ]; then
    echo "setup.py has version $setup_py_version, not $version."
    echo "Aborting."
    exit 1
fi

# Check working directory is clean
if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "You have changes that have yet to be committed."
    echo "Aborting PyPI upload and attempting to commit your changes."
    scripts/commit.sh
fi

# Check that we are on main
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "main" ]; then
    echo "You are on $branch but should be on main for releases."
    echo "Aborting."
    exit 1
fi

# Create and push git tag
git tag -m "Version v$version" "v$version"
git push --tags


# Create and publish package
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
rm -r dist && rm -r build && rm -r bio_check.egg-info

echo "Version v$version has been published on PyPI and has a git tag."

echo "$version" > ./bio_check/_VERSION