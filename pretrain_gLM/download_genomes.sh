#!/bin/sh

# This script is for downloading reference genomes
cd download
echo "Downloading reference genomes..."
wget -c -i download_urls.txt
echo "Download complete."
cd ..