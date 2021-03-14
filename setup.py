import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

try:
    setuptools.setup(
        name = 'mathreader-training',
        version = '0.1',
        author = 'Caroline Reis',
        author_email = 'caroline.reis@rede.ulbra.br',
        long_description = long_description,
        url = 'https://github.com/carolreis/mathreader-training',
        packages = setuptools.find_packages()
    )
except Exception as e:
    pass
