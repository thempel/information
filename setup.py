from setuptools import setup

setup(name='informant',
      version='0.1',
      description='Minimal package containing estimators for mutual and directed informant',
      url='https://github.com/thempel/information.git',
      packages=['informant'],
      zip_safe=False,
      install_requires=['numpy'],
      test_suite='py.test',
      tests_require=['pytest'])
