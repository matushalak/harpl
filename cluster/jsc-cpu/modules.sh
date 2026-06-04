module purge
module load Stages/2025
module load GCC

# Base scientific Python modules recommended by the JSC sc_venv_template.
module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
module load Flask Seaborn
module load scikit-learn tensorboard h5py
