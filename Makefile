clean:
	-rm -r build dist osds.egg-info

dist:
	python setup.py sdist bdist_wheel

test:
	pytest test/test_utils.py

pip:
	twine upload --repository pypi dist/*
