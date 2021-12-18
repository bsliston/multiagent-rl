from setuptools import setup, find_packages

setup(
    name="multiagent-rl",
    version="1.0",
    description="A multi-agent reinforcement learning training module",
    author="Brandon Liston",
    author_email="liston0456@aol.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "gym>=0.21.0",
        "pettingzoo[all]",
        "torch",
    ],
)
