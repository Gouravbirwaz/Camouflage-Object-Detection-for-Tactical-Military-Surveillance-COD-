from setuptools import setup, find_packages

setup(
    name="cod_project",
    version="0.1.0",
    description="Camouflaged Object Detection using Deep Gradient Networks (DGNet)",
    author="COD Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "pyyaml>=6.0",
    ],
)
