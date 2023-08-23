install:
	pip install -r API/Flask/requirements.txt

format:
	black API/Flask/*.py train/*.py
run:
	python API/Flask/app.py
pylint:
	pylint --disable=R,C API/Flask/*.py train/*.py
make:
	make install
	make format
	make run
