from setuptools import setup, find_packages
import qute

setup(name="qute",
      version=qute.__version__,
      author="Aaron Ponti",
      author_email="aaron.ponti@bsse.ethz.ch",
      url="https://github.com/aarpon/qute",
      description="Leverages and extends several PyTorch-based framework and tools.",
      packages=find_packages(),
      provides=["qute"],
      license="Apache-2.0",
      classifiers=["Development Status :: 2 - Pre-Alpha",
                   "Intended Audience :: Education",
                   "Natural Language :: English",
                   "Operating System :: POSIX :: Linux",
                   "Operating System :: MacOS :: MacOS X",
                   "Operating System :: Microsoft :: Windows",
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
                   "Topic :: Education"
                  ],
      install_requires=[]  # Please create the environment with `conda env create -f environment.yml` first
)
