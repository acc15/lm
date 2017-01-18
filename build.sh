#!/bin/sh

git submodule init 
git submodule update

mkdir -p build
cd build
cmake ../
cmake --build .
./lm_test
