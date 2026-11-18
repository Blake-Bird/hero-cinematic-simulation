import re
from pathlib import Path

from setuptools import find_packages, setup

# Single source of truth for the version: StreaMAX/__init__.py.  It cannot be
# imported here (it pulls in JAX), so read it out of the file instead. Keeping
# the number in one place is deliberate -- setup.py and __init__.py drifted
# apart before, and a version bumped in only one of them produced a PyPI
# release whose contents did not match its version number.
INIT = Path(__file__).parent / "StreaMAX" / "__init__.py"
VERSION = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', INIT.read_text(), re.M).group(1)

setup(
    name="StreaMAX",
    version=VERSION,
    author="David Chemaly",
    author_email="dc824@cam.ac.uk",
    description="A JAX-accelerated stellar stream generator (StreaMAX)",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/David-Chemaly/StreaMAX",
    project_urls={
        "Source": "https://github.com/David-Chemaly/StreaMAX",
        "Inference pipeline": "https://github.com/David-Chemaly/StreaMAX-fit",
    },
    license="MIT",
    packages=find_packages(exclude=["notebooks*", "tests*"]),
    install_requires=[
        "jax>=0.4.0",
        "numpy>=1.24",
        "astropy>=5.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    keywords=["astronomy", "stellar streams", "dark matter", "jax"],
    python_requires=">=3.10",
    include_package_data=True,
)
