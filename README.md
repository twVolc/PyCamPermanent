# PyCam project repo

## Pre-requsities

- Windows 10/11
- Conda installed

## Installation instructions

1. Clone the `standalone` branch of the repo and move into the PyCamPermanent directory:
```
git clone -b standalone https://github.com/ubdbra001/PyCamPermanent.git
cd PyCamPermanent
```

2. Create a new `python 3.9` conda environment and activate it:
```
conda create -n pycam_standalone python=3.9
conda activate pycam_standalone
```

3. Install dependencies from the requirements.txt file:
```
pip install -r requirements.txt
```

4. Run `pycam`:
```
python run_pycam.py
```
