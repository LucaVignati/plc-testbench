import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='alohaecc',
                 version='0.0.1',
                 author='Christopher Walker',
                 author_email='chris@elk.audio',
                 description="A prototypical Error Concealment \
                     bench test for Aloha",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 install_requires=[
                    'numpy',
                    'soundfile',
                 ],
                 )
