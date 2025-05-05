from setuptools import find_packages, setup

setup(
    name='flood_model',
    version='0.1.0',
    description='Flood prediction using ConvLSTM models',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'rasterio',
        'tqdm',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'flood-train = flood_model.train:main',
            'flood-validate = flood_model.validate:main',
        ],
    },
)