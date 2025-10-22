from setuptools import setup, find_packages
from pathlib import Path

# Read the README.md for long_description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='agensgraph-python',
    version='1.0.2',
    description='Psycopg2 type extension module for AgensGraph',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['psycopg2>=2.5.4'],

    packages=find_packages(exclude=['tests']),
    test_suite = "tests",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Database',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],

    author='Umar Hayat',
    author_email='skaisw@skaiworldwide.com',
    maintainer='Umar Hayat',
    maintainer_email='skaisw@skaiworldwide.com',
    url='https://github.com/skaiworldwide-oss/agensgraph-python',
    license='Apache License Version 2.0',
    keywords='agensgraph graph database postgresql driver',
)
