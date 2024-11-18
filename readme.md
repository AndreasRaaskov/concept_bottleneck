# CUB concept bottleneck

This project aims to replicate some of the results in paper [Concept bottleneck](https://proceedings.mlr.press/v119/koh20a/koh20a.pdf).

This code only paper related to concept bottleneck model trained on the [Caltech-UCSD Birds-200-2011 dataser](https://www.vision.caltech.edu/datasets/cub_200_2011/) it does not yet support many of the experiment in the paper, however the code offer more flexibility than the original code.

I started out with [original code base](https://github.com/yewsiang/ConceptBottleneck?tab=readme-ov-file) but have rewritten most of the code base. 

The rewrite serve two purpose, first I found several things in the original code I believe to be a mistake and wanted to understand the code in details. The second part is that this code could be used as an educational resource and thus I have tried to improve readability a lot compered to the original code and simplified a lot of proses, as well as adding Hydra to easily setup and track experiments.

This code base also contains scripts to run the code on DTU HPC system.

I will activly maintain this code until December 2024. If you have any questions or discovers a bug you can mail me at s183901@dtu.dk


## Installation

### Prerequisites

- Python 3.9 to 3.12 
- pip (Python package installer)

You can check your Python version by typing:

   ```
   python -V
   ```





#### On Linux/macOS and DTU hpc:

 Create a virtual environment:
   ```
   python -m venv env
   ```

Activate the virtual environment:
   ```
   source env/bin/activate
   ```

#### On Windows:


Create a virtual environment:
   ```
   python -m venv env
   ```
 Activate the virtual environment:
   ```
   .\env\Scripts\activate
   ```

### Installing Dependencies

Once your virtual environment is activated, install the required packages:

```
pip install -r requirements.txt
```

If you wish to use a GPU get the command for installing torch for CUDA on: https://pytorch.org/get-started/locally/

If you are usure of your CUDA version type
```
nvidia-smi
```



## Usage


## Notable changes from original code.
None