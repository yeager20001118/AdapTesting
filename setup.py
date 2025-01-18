from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="adaptesting",  # Your package name
    version="0.0.1",
    author="Xunye Tian",
    author_email="xunyetian.ml@gmail.com",
    description="A toolbox to directly access to various two-sample testing methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # url="",
    install_requires=[
        "torch",
        "typing",
        "jax",
        "pytorch_tabnet",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extra_require={
        "dev": [
            "twine",
        ]
    },
    python_requires=">=3.9",
)
