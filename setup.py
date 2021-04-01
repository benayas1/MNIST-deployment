import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(where=".")
print(packages)

setuptools.setup(
    name="mnist_demo", # Replace with your own username
    version="0.0.1",
    author="Alberto Benayas",
    author_email="benayas1@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/benayas1/MNIST-deployment",
    project_urls={
        "Bug Tracker": "https://github.com/benayas1/MNIST-deployment",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=packages,
    python_requires=">=3.6",
)