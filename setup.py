from setuptools import setup, find_packages
# import yaml

__version__ = "0.1.0"
__name__ = "AnimalsVision"
__status__ = "development"

setup(
    name=__name__,
    version=__version__,
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "AnimaslV=project.cli:cli"
        ]
    }
)
