"""Setup for FastDETR."""

from setuptools import setup, find_packages

setup(
    name='fastdetr',
    version='1.0.0',
    description='Accelerated Detection Transformer with Multi-Scale Deformable Attention',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='FastDETR Contributors',
    license='Apache 2.0',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'Pillow>=9.0.0',
        'tqdm>=4.60.0',
        'PyYAML>=6.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
