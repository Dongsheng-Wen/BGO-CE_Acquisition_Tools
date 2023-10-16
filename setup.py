from setuptools import setup



setup(
    name='BGO_scripts',
    version='v0.1.0',
    license ='MIT',
    author='Dongsheng Wen',
    author_email='wen94@purdue.edu',
    description='local useful scripts for BGO.',
    packages = ['bgotools'],
    platforms='any',
    install_requires=[
    'numpy >= 1.13.3',
    'matplotlib >= 2.1.0',
    'pandas >= 0.23.4',
    'scipy >= 1.5.1']
)

