#!/usr/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake based on a separate PETSc         #
# installation which uses Mmg and ParMmg and custom mpich installations. #
#                                                                        #
# Note that we use custom PETSc and Firedrake branches.                  #
# ====================================================================== #

# Create installation directory
tmp=$(pwd)
mkdir $HOME/software
export SOFTWARE=$HOME/software  # Modify as appropriate
cp ./* $SOFTWARE
cd $SOFTWARE

# Install PETSc and Firedrake
./install_petsc_custom_mpich.sh
./install_firedrake_honour_petsc_custom_mpich.sh

# Cleanup
cd $tmp
for f in .; do
	rm $SOFTWARE/$f
done
