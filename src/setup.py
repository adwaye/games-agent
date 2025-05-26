

from setuptools import setup

setup(
    name='games_agent',
    version='0.0.1',
    description='A useful module',
    author='Man Foo',
    author_email='adwayerambojun@gmail.com',
    packages=['games_agent'],  # same as name
    # external packages as dependencies
    install_requires=['wheel', 'bar', 'greek'],
    # scripts=[
    #          'scripts/cool',
    #          'scripts/skype',
    #         ]
)
