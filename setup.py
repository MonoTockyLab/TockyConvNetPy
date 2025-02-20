from setuptools import setup, find_packages

setup(
    name='TockyConvNetPy',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'TockyConvNetPy': [
            'data/**/*',
            'assets/*',
            'notebooks/*',
        ],
    },
    description='Machine Learning Methods for Flow Cytometric Fluorescent Timer Data (Tocky Data)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Masahiro Ono',
    author_email='monotockylab@gmail.com',
    url='https://github.com/MonoTockyLab/TockyConvNetPy',
    install_requires=[
        'tensorflow==2.10.0',
        'keras==2.10.0',
        'numpy>=1.21.0,<1.22.0',
        'pandas>=1.5.3,<1.6.0',
        'scipy>=1.10.1,<1.11.0',
        'scikit-learn>=1.1.2,<1.2.0',
        'matplotlib>=3.5.3,<3.6.0',
        'scikit-image>=0.19.2,<0.20.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    python_requires='>=3.7',

)
