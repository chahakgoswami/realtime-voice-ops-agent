from setuptools import setup, find_packages

setup(
    name="voice_ops_agent",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "voice-ops=voice_ops_agent.__main__:main",
        ],
    },
)
