init:
	pip install -U -r requirements.txt

work: init
	./analyze.py
	./reference.py
