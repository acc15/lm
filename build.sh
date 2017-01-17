#!/bin/sh

mkdir -p build
cd build
cmake ../
cmake --build .
./lm_test
