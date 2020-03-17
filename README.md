# SIGMOD20 Programming Contest | Quick start package for Entity Resolution

The Quick Start Package for Entity Resolution could be used as a starting code base for the SIGMOD 2020 Programming Contest.

## Prerequisites

- Python 3.*
- Pip

## Installing

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Install virtualenv to create an isolated environment:
```
$ pip install virtualenv
```

Create a new virtual environment named venv and activate it:
```
$ virtualenv venv
$ source venv/bin/activate
```

N.B. Usage of virtualenv is not mandatory but recommended.

Install the requirements:

```
$ pip install -r requirements.txt
```

## Running

Run the project:
```
$ python main.py
```

This command will produce a CSV file (the submission) in the output directory ("outputh_path") and will print intermediate results in the shell.

N.B. The program will be executed on a mock dataset. If you want to change dataset just edit the value of the "dataset_path" variable.