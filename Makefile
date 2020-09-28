clean:
	-rm -r build dist osds.egg-info

dist:
	python setup.py sdist

test:
	pytest test/test_utils.py

pip:
	python setup.py sdist bdist_wheel
	python3 -m twine upload --repository pypi dist/*