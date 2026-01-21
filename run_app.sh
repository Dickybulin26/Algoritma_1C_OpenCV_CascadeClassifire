#!/bin/bash
# Script untuk menjalankan aplikasi Flask dengan virtual environment

# check python version
python --version

# Check installed packages
pip list

# install python virtual environment for windows, linux and macos
pip install virtualenv

# make virtual environment
python3 -m venv venv

# Install dependencies
pip install -r requirements.txt

# Aktifkan virtual environment dan jalankan aplikasi
venv/bin/python app.py
