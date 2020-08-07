# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:11:32 2020

@author: meppenga
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    
import os
MainPath, _= os.path.split( os.path.abspath(__file__))
MainPath = os.path.join(MainPath,'MaskRCNN')
dirs = ['Data','Data\\Cache','Data\\Cache\\Rotation','Data\\Cache\\Rotation\\Validation', 'Data\\Cache\\Rotation\\Training',
        'Data\\Cache\\Mirror','Data\\Cache\\Mirror\\Validation','Data\\Cache\\Mirror\\Training',
        'Data\\Examples','Data\\Inference','Data\\Training','Data\\Training\\Results']
for path in dirs:
    if not os.path.isdir(os.path.join(MainPath, path)):        
        os.mkdir(os.path.join(MainPath, path))

GitPath = os.path.join(MainPath,'Configurations')
with open(os.path.join(GitPath,'GitInstallPath.py'),'w') as obj:
    obj.write('# This is a code generated file\n')
    obj.write('# It is generated when one runs the Setup.py file\n')
    obj.write('# Do not change it content\n')
    obj.write('GitPath = "'+GitPath+'"')





setup(
        name='CellDetection',
        version='1.1',
        url='https://github.com/Brinkslab/ProteinMLDetection',
        author='MartijnEppenga',
        author_email='martijn.eppenga@hotmail.com',
        license='MIT',
        description='Mask R-CNN used for cell detection within microscope images',
        packages= ["MaskRCNN.Configurations","MaskRCNN.DataGenerators","MaskRCNN.Engine","MaskRCNN.Miscellaneous"],
        include_package_data=True,
        python_requires='>=3.4',
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Scientific/Engineering :: Image Segmentation",
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        keywords="image instance segmentation object detection mask rcnn r-cnn tensorflow keras",
    )    


