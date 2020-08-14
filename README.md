# General

This repository contains the code for my numerical experiments on the 
gym data during my Master's Thesis. 

## General Usage

This repository contains both convenience functions for operating on the data 
as well as runnable scripts that compute different sorts of statistics 
on the data. Before everything, it is advisable to create a new virtual 
environment to avoid cluttering your system's native Python distribution. 

### Pipenv

This repository contains a Pipfile for usage with Pipenv; for more information, 
see the [Pipenv docs](https://pipenv-fork.readthedocs.io/en/latest/).

Installation on macOS is as easy as

```
brew install pipenv
```
provided you have Homebrew installed on your Mac. Afterwards, dependencies can
be installed from the Pipfile by simply running
```
pipenv install
```
in the top-level directory. The virtual environment can be activated by the 
```
pipenv shell
```
command. 


### Cython distribution

On top of the Python code, this repository also contains some functions written
in Cython to speed up computations on the data. Read about Cython usage 
[here](https://cython.readthedocs.io/en/latest/). The compilation and 
installation process is handled by Python's `setuptools` package. For 
installation, from the top level directory complete these steps while inside
your virtual environment:

```
cd regularity_analysis
python setup.py build_ext --inplace
```


## Running statistics experiments 

Some experiments on the dataset are located inside the `bin` folder. These are 
based on a config file architecture: Command line arguments for statistical 
functions and interesting hyperparameters for these functions can be specified
inside these yaml files. Upon running these scripts, the yaml files are then 
envoked and parsed by the PyYaml module. Some sample configuration files are
located inside the `cfg` directory, though you can pass your own to the scripts
by supplying the --config_file flag. 

Running a script takes one line. From the root of the repository, run
```
python -m bin.<script_name>
```
*Notice that the file is run as a module, so no .py should be added at the end.
For example, for running the `changepoint_detection` script file inside bin, 
you should type

```
python -m bin.changepoint_detection
```

from the root directory. If you are working on the repository yourself, a
local installation could make this process easier, I did not figure out in time
how to do that. 

### Other useful information

The code leverages mostly pandas and numpy/scipy, as the dataset is small 
enough to fit into memory. Some data preparation code can be found inside the
Jupyter Notebooks, for those, just fire up a Jupyter Lab instance. 
