from setuptools import setup, find_packages

setup(
    name="particle_aggregation_model",
    version="0.1.0",
    description="A model for calculating particle aggregation based on discrete aggregate size classes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Laurin Steidle",
    author_email="laurin.steidle@uni-hamburg.de",
    url="https://github.com/465b/particle_aggregation_model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy"
    ],
)