from setuptools import setup, find_packages

setup(
    name="CAST",
    version="0.3",
    packages=find_packages(),
    install_requires=[
        'torch',
        'dgl',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'h5py',
        'statsmodels',
        'tqdm',
        'geopandas',
        'Rtree',
        'scanpy',
        'libpysal',
        'ipython',
        'jupyterlab',
        'jupyter',
        'numpy',
        'pandas'
    ]
)
