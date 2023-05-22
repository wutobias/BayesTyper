from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name='BayesTyper',
    author='Tobias Huefner',
    author_email='tobias.huefner@biophys.mpg.de',
    description="A package for fitting forcefield parameter type definitions.",
    version=__version__,
    license='MIT',
    platforms=['Linux'],
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True
    )
