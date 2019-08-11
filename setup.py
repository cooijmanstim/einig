import os
import setuptools

setuptools.setup(
  name='eigen',
  version='0.0.1',
  packages=setuptools.find_packages(),
  author='Tim Cooijmans',
  author_email='cooijmans.tim@gmail.com',
  description='Tensorflow einsum generalizations',
  long_description=open(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'README.md')).read(),
  license='BSD 3-clause',
  url='http://github.com/cooijmanstim/eigen',
  install_requires="numpy tensorflow".split(),
  python_requires=">=3",
  classifiers=['Development Status :: 3 - Alpha',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: BSD License',
               'Operating System :: OS Independent',
               'Topic :: Scientific/Engineering'],
)
