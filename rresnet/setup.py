import setuptools

setuptools.setup(
    name="rresnet",
    version="1.1.0",
    author="Isay Katsman",
    author_email="isay.katsman@yale.edu",
    description="Riemannian Residual Neural Networks with PHDM input maps",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.26", "torch>=2.1"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
