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

# handle existing version
function get_version {
   python -c "import toml; print(toml.load('pyproject.toml')['tool']['poetry']['version'])"
}

current_version=$(get_version)
if [ "$current_version" != "$version" ]; then
    echo "pyproject.toml has version $current_version, not $version."
    echo "Aborting."
    exit 1
else
    echo "Version matches: $current_version"
fi

# Check working directory is clean
if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "You have changes that have yet to be committed."
    echo "Aborting PyPI upload and attempting to commit your changes."
    ../../commit.sh
fi

# Check that we are on main
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "main" ]; then
    echo "You are on $branch but should be on main for releases."
    echo "Aborting."
    exit 1
fi

# update internal version
echo "$version" > ./bio_check/_VERSION

# Create and push git tag
git tag -m "Version v$version" "v$version"
git push --tags

# Create and publish package
function get_pypi_token {
  cat ~/.ssh/.bio-check-pypi
}

pypi_token=$(get_pypi_token)
poetry build
poetry publish --username __token__ --password "$pypi_token"
rm -r dist

# If using a non-poetry build
# python setup.py sdist bdist_wheel
# twine check dist/*
# twine upload dist/*
# rm -r dist && rm -r build && rm -r bio_check.egg-info

echo "Version v$version has been published on PyPI and has a git tag."



