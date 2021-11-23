from setuptools import setup, find_packages

setup(
    name="mustacher",
    version="0.0.1",
    packages=find_packages(include=['mustacher', 'mustacher.*']),
    include_package_data=True
)
