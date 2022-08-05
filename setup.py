from setuptools import setup

setup(name='informant',
      version='0.1',
      description='Minimal package containing estimators for mutual and directed informant with MSM probabilities',
      url='https://github.com/markovmodel/information.git',
      packages=['informant'],
      zip_safe=False,
      install_requires=['numpy', 'msmtools', 'tqdm', 'six', 'pathos'],
      test_suite='py.test',
      tests_require=['pytest'])
