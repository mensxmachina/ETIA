from setuptools import setup, find_packages

setup(
    name='ETIA',
    version='1.0.3',
    packages=find_packages(include=['ETIA', 'ETIA.*']),
    author_email='droubo@csd.uoc.gr',
    description='Automated Causal Discovery Library',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mensxmachina/ETIA',
    include_package_data=True,
    package_data={
        'ETIA.AFS.feature_selectors': ['fbed_with_idx.R', 'ses_with_idx.R'],
        'ETIA.CausalLearning.CausalLearning.algorithms.jar_files': ['*.jar'],
        'ETIA.CRV.adjustment': ['*.R'],
    },
    install_requires=[
        'causalnex',
        'cdt',
        'joblib',
        'jpype1',
        'networkx>=3.2.1',
        'numpy',
        'pandas',
        'pgmpy',
        'py4cytoscape',
        'pywhy-graphs',
        'scikit-learn',
        'tigramite>=5.2.3.1',
        'matplotlib',
        'dcor'
    ],
)
