#!/bin/bash
isort --profile black -l 79 gevidaq
black -l 79 gevidaq
line_length=E501
unused_var=F841
flake8 --extend-ignore E203,$line_length,$unused_var gevidaq
