from setuptools import setup, find_packages

setup(
    name="liveaudio",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time audio processing library based on librosa",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
	install_requires=[
		"numpy>=2.2.4",
		"numba>=0.61.2",
		"scipy>=1.15.2",
		"librosa>=0.11.0",
	],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
        ],
    },
)