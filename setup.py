from setuptools import setup, find_packages

setup(
    name='tamnun',
    version="0.1.0",
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
    ]
)
