from setuptools import setup
import toml
pyproject = toml.load("pyproject.toml")

setup(name=pyproject["project"]["name"],
      version=pyproject["project"]["version"],
      description=pyproject["project"]["description"],
      url=pyproject["project"]["urls"]["repository"],
      packages=['informant'],
      zip_safe=False,
      install_requires=pyproject["project"]["dependencies"],
      )
