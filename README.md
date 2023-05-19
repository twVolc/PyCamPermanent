# PyCam project repo

## Pre-requsities

- Windows 10/11
- `conda` installed ([installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html))
- `git` installed (optional but recommended; [installation instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))

## Installation instructions

1. If you have `git` installed then clone the `standalone` branch of the repo and move into the PyCamPermanent directory:
```
git clone -b standalone https://github.com/ubdbra001/PyCamPermanent.git
cd PyCamPermanent
```

- If you don't have `git` installed you can still download and extract [the repository](https://github.com/ubdbra001/PyCamPermanent/tree/standalone) (for instructions, see 'If All Else Fails, Get The Code' on [this page](https://www.howtogeek.com/827348/how-to-download-files-from-github/))
    - Once downloaded you need to extract the files from the zipped file and navigate to the newly extracted directory in your terminal

2. Create a new `python 3.8` conda environment and activate it:
```
conda create -n pycam_standalone python=3.8
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
