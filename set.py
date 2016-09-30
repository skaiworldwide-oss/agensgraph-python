from setuptools import setup, find_packages
setup(
    name = "agtype",
    version = "1.0",
    scripts = ["agtype.py"],
    install_requires=["psycopg2>=2.5.4"],
    test_suite = "test_agtype",
    author = "Kyungtae Lee",
    author_email = "ktlee@bitnine.net",
    description = "A type extension module for agens-graph",
    url = "http://bitnine.net"
)
