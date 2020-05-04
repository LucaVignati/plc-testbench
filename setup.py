import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='alohaecc',
                 version='0.0.1',
                 author='Christopher Walker',
                 author_email='chris@elk.audio',
                 description="A prototypical Error Concealment
                              bench test for Aloha",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://bitbucket.org/mindswteam/alohaecc"
                 packages=setuptools.find_packages(),
                 classifiers=[
                     "Programming Language :: Python :: 3",
                 ]
                 install_requires=[
                    'numpy',
                    'soundfile',
                 ],
                 python_requires='>=3.7',
                 )
