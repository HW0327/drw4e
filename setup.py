import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="drw4e",                     # This is the name of the package
    version="0.0.13",                        # The initial release version
    author="Haoyu Wang",                     # Full name of the author
    author_email="haoyuwoody0327@gmail.com",
    url="https://github.com/HW0327/drw4e",
    description="A Python Package dedicated to estimate brightness variation characteristics of quasars",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=['drw4e'],    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    install_requires=['numpy', 'scipy']       # Install other dependencies if any
)

