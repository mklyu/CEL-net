#!/bin/bash
python3 -m venv venv

source ./venv/bin/activate

pip install --upgrade pip
pip3 install --upgrade pip

pip3 install -r pip-requirements.txt