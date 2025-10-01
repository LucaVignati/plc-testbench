import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

exec(open("plctestbench/__version__.py").read())

setuptools.setup(name='plc-testbench',
                 version=__version__,
                 author='Marcel Ellert',
                 author_email='marcel_ellert@web.de',
                 description="Framework comparison and benchmarking of error concealment algorithms",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://bitbucket.org/mindswteam/ecc-testbench",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     "Programming Language :: Python :: 3",
                 ],
                 install_requires=[
                    'anytree==2.8.0',
                    'Brian2==2.8.0.4',
                    'brian2hears==0.9.2',
                    'click==8.2.1',
                    'essentia==2.1b6.dev1110',
                    'Flask==2.2.5',
                    'ipykernel==6.29.5',
                    'ipython==8.12.3',
                    'ipywidgets==8.1.6',
                    'librosa==0.9.2',
                    'matplotlib==3.10.3',
                    'notebook==7.4.1',
                    'numpy==1.24.3',
                    'pandas==2.3.0',
                    'plotly==6.0.1',
                    'pybind11==2.10.4',
                    'pymongo==4.3.3',
                    'ruamel.yaml==0.18.10',
                    'scipy==1.16.0',
                    'seaborn==0.13.2pip',
                    'setuptools==65.5.0',
                    'soundfile==0.13.1',
                    'statsmodels==0.14.4',
                    'tensorflow==2.11.0',
                    'tinydb==4.8.2',
                    'tinyrecord==0.2.0',
                    'tqdm==4.67.1',
                    'typing-inspect==0.9.0',
                 ],
                 python_requires='==3.10.17',
                 )
