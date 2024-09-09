import os


try:
    from .version import __version__
except ImportError:
    path = os.path.join(os.path.dirname(__file__), 'version.txt')
    if os.path.isfile(path):
        with open(path) as stream:
            __version__ = stream.readline().strip()
    else:
        __version__ = 'failed-to-get-version'
