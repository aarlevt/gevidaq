#!/bin/bash
isort --profile black -l 79 .
black -l 79 .
flake8 --extend-ignore E203,E501 .
