#!/usr/bin/env bash


# The following script serves as a utility for installing this repository with the Smoldyn requirement on a Silicon Mac

# set installation parameters
dist_url=https://www.smoldyn.org/smoldyn-2.73-mac.tgz
tarball_name=smoldyn-2.73-mac.tgz
dist_dir=${tarball_name%.tgz}

# uninstall existing version
conda run -n server pip-autoremove smoldyn -y || return

# download the appropriate distribution from smoldyn
wget $dist_url

# extract the source from the tarball
tar -xzvf $tarball_name

# delete the tarball
rm $tarball_name

# install smoldyn from the source
cd $dist_dir || return

if conda run -n server sudo -H ./install.sh --force-reinstall ; then
  cd ..
  # remove the smoldyn dist
  rm -r $dist_dir
  echo "Smoldyn successfully installed. Done."
else
  echo "Could not install smoldyn"
fi
