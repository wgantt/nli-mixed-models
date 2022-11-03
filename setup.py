from setuptools import find_packages, setup

setup(
    name="nli_mixed_models",
    version="0.1",
    description="Mixed Effects Models for Natural Language Inference",
    url="",
    author="Will Gantt",
    author_email="wgantt@cs.rochester.edu",
    license="MIT",
    packages=find_packages(),
    package_dir={"nli_mixed_models": "nli_mixed_models"},
    install_requires=[
        "numpy==1.20.0",
        "pandas==1.0.5",
        "torch==1.3.1",
        "torchvision==0.4.*",
        "fairseq==0.9.*",
    ],
    test_suite="nose.collector",
    tests_require=["nose"],
    zip_safe=False,
)
