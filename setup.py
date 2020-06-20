from setuptools import find_packages, setup

setup(name='nli_mixed_models',
      version='0.1',
      description='Mixed Effects Models for Natural Language Inference',
      url='',
      author='Aaron Steven White',
      author_email='aaron.white@rochester.edu',
      license='MIT',
      packages=find_packages(),
      package_dir={'nli_mixed_models': 'nli_mixed_models'},
      install_requires=['numpy==1.17.*',
                        'pandas==0.25.*',
                        'torch==1.3.*',
                        'torchvision==0.4.*',
                        'fairseq==0.9.*'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)