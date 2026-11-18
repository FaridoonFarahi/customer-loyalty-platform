pipeline:
	python -m src.pipeline

test:
	python -m pytest

lint:
	ruff check src tests streamlit_app.py

score:
	python -m src.predict --start-day 712 --campaign-type TypeA --top-share 0.2

api:
	uvicorn src.api:app --reload

app:
	streamlit run streamlit_app.py

docker:
	docker build -t loyalty-scoring .

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db

.PHONY: pipeline test lint score api app docker mlflow-ui
