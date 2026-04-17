install:
	pip install -r requirements.txt

format:
	black model_pipeline.py main.py

quality:
	flake8 model_pipeline.py main.py

security:
	bandit -r model_pipeline.py main.py

prepare:
	python3 main.py --prepare

train:
	python3 main.py --train

evaluate:
	python3 main.py --evaluate

all:
	python3 main.py --all

ci: format quality security

run: prepare train evaluate

api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

docker-build:
	docker build -t hadil_lajili_wiem_saafi_4ds2_mlops .

docker-run:
	docker run -d -p 8000:8000 --name churn_api hadil_lajili_wiem_saafi_4ds2_mlops

docker-push:
	docker push lajilihadil/hadil_lajili_wiem_saafi_4ds2_mlops

