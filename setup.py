from setuptools import setup, find_packages


def parse_reqs(path):
    """Read requirements, skipping -r includes and blank/comment lines."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-r"):
                lines.append(line)
    return lines


setup(
    name="sketch2scene",
    version="1.0.0",
    author="Sagnik Chandra",
    author_email="sagnikchandra027@gmail.com",
    description="Multimodal decoder-only transformer: sketch → scene description + image completion",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sagnik120/sketch2scene",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    python_requires=">=3.10",
    install_requires=parse_reqs("requirements.txt"),
    extras_require={
        "dev": parse_reqs("requirements-dev.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)