* python setup.py bdist_wheel -> build the whl
* python setup.py sdist -> build the tar.gz
* pip install -e . -> install the package locally
* twine upload -r testpypi dist/* -> upload distributions to test.pypi.org
* twine upload dist/* -> upload distributions to pypi.org (official)