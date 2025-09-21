.PHONY: train eval api test

train:
\tpython -m src.train

eval:
\tpython -m src.evaluate

api:
\tuvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
\tpytest -q
