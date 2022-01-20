import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("ecc_external",
                      ["bindings/external_ecc_bindings.cpp"])
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='ecc-testbench',
                 version='0.0.1',
                 author='Christopher Walker',
                 author_email='chris@elk.audio',
                 description="A prototypical Error Concealment \
                              bench test for Aloha",
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
                 ],
                 python_requires='>=3.7',
                 ext_modules=ext_modules,
                 cmdclass={"build_ext": build_ext}
                 )
