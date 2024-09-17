import subprocess
from pathlib import Path
from setuptools import find_packages, setup


def get_version():
    path = Path(__file__).absolute().parent / 'rirtk' / 'version.txt'
    assert path.is_file(), f'File {path} does not exist'
    return path.read_text().strip()


def get_version_sha():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        sha = sha[:5]
    except subprocess.CalledProcessError:
        sha = 'failed-to-get-sha'
    return f'{get_version()}.{sha}'


path = Path(__file__).absolute().parent / 'rirtk' / 'version.py'
path.write_text(f"__version__ = '{get_version_sha()}'\n")


with open('README.md', encoding='utf-8') as stream:
    long_description = stream.read()


setup(
    name='rir-classifier',
    version=get_version(),
    python_requires='>=3.8',
    author='Yuri Khokhlov',
    author_email='khokhlov@speechpro.com',
    description='RIR Classifier',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/speechpro/rir-classifier',
    project_urls={
        'Bug Tracker': 'https://github.com/speechpro/rir-classifier/issues',
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: POSIX :: Linux',
    ],
    install_requires=[
        'inex-launcher',
    ],
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    keywords='speechpro rir rirtk rir-classifier',
)
