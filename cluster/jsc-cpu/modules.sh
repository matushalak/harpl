module --force purge
module load Stages/2026
module load GCC

# Base scientific Python modules recommended by the JSC sc_venv_template.
module load numba tqdm matplotlib IPython SciPy-Stack bokeh git
module load Flask Seaborn
module load scikit-learn tensorboard h5py
