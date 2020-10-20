import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="karl-scheduler", # Replace with your own username
    version="0.0.1",
    author="Shi Feng",
    author_email="sjtufs@gmail.com",
    description="Scheduler for karl.qanta.org",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pinafore/fact-repetition",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)