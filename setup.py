from setuptools import setup

setup(name='information',
      version='0.1',
      description='Minimal package containing estimators for mutual and directed information',
      url='https://github.com/thempel/information.git',
      packages=['information'],
      zip_safe=False,
      install_requires=['numpy'],
      test_suite='py.test',
      tests_require=['py.test'])
