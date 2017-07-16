from setuptools import setup

setup(name='kagglelib',
      version='0.1',
      description='Library of most useful functions for Kaggle and other machine learning competitions',
      url='https://github.com/Arcady27/KaggleLib',
      author='Arkadiy Dushatskiy',
      author_email='arcady27@gmail.com',
      license='MIT',
      packages=['kagglelib', 
      'kagglelib.feature_engineering', 
      'kagglelib.model_predicting', 
      'kagglelib.model_ensembles',
      'kagglelib.model_training', 
      'kagglelib.model_tuning',
      'kagglelib.feature_selection',
      'kagglelib.preprocessing',
      'kagglelib.utils',
      'kagglelib.Model'
      ],
      zip_safe=False)