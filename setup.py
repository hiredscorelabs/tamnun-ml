import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

VERSION = '0.1.0'

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(
    name='tamnun',
    version=VERSION,
    description="An easy to use open-source library for advanced Deep Learning and Natural Language Processing",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='Deep Learning Natural Language Processing NLP Machine Learning Transfer Learning',
    license='Apache',
    packages=find_packages(exclude=['test*']),
    install_requires=[
        'numpy==1.15.4',
        'scikit-learn==0.20.2',
        'torch==1.1.0',
        'pytorch-transformers',
    ],
    cmdclass = {
        'verify': VerifyVersionCommand,
    }
)
