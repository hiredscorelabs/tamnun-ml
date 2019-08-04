clean:
	rm -rf dist/*

dev:
    pip install -r dev-requirements.txt -e .

package:
	python setup.py sdist
	python setup.py bdist_wheel

test:
	py.test tests