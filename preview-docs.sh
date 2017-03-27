#/!/bin/sh

rm -rf build/doc
doxygen

google-chrome-stable build/doc/html/index.html