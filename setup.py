import setuptools

with open("README.md", "r", encoding="utf8") as fh:
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
                    'pymongo',
                    'tqdm',
                    'matplotlib',
                    'tensorflow',
                    'cpp-plc-template',
                    'burg-plc',
                 ],
                 python_requires='>=3.7',
                 )
