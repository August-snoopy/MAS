# 安装requirements.txt中的依赖包
# pip install -r requirements.txt
# 运行setup.py
# python setup.py

from setuptools import setup, find_packages

setup(
    name='data_mini',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'data_mini = data_mini.post_process:main'
        ]
    }
)
