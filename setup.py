import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("plc_external",
                      ["../ecc_python_bindings/ecc_python_bindings/python_bindings.cpp"])
]

with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open("plctestbench/__version__.py").read())

setuptools.setup(name='plc-testbench',
                 version=__version__,
                 author='Luca Vignati',
                 author_email='luca.vignati@vignati.net',
                 description="Framework comparison and benchmarking \
                              of error concealment algorithms",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://bitbucket.org/mindswteam/ecc-testbench",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     "Programming Language :: Python :: 3",
                 ],
                 install_requires=[
                    'numpy',
                    'soundfile',
                    'anytree',
                    'pybind11',
                    'pymongo'
                    'tqdm',
                    'matplotlib',
                    'tensorflow',
                 ],
                 python_requires='>=3.7',
                 ext_modules=ext_modules,
                 cmdclass={"build_ext": build_ext}
                 )
