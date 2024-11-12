pull:
	dvc pull

fastrun:
	# train simple mode
	# keep in report
	HYDRA_FULL_ERROR=1 python src/train.py trainer.max_epochs=3 trainer.precision=16

train:
	# keep in assets
	HYDRA_FULL_ERROR=1 python src/train.py -m
	echo "Best Hparams"
	cat multirun/*/*/optimization_results.yaml
	echo "pushing to S3"
	aws s3 cp multirun/ s3://abhiya-emlo-bucket/training-$$(date +"%m-%d-%H%M%S")/ --recursive

eval:
	HYDRA_FULL_ERROR=1 python src/eval.py 

inference:
	rm -rf samples/outputs/*
	HYDRA_FULL_ERROR=1 python src/inference.py --input_folder samples/inputs/ --output_folder samples/outputs/ --ckpt_path samples/checkpoints/epoch_019.ckpt 

test:
	pytest --cov --cov-report=xml
	coverage run -m pytest
	# setup code-coverage
	# coverage xml -o coverage.xml
  
trash:
	pwd
	rm -rf data/dogs_dataset
	find . -type d \( -name '__pycache__' -o -name 'logs' -o -name 'outputs' -o -name 'multirun' \) -exec rm -rf {} +
	rm -rf samples/outputs/*
	rm assets/test_confusion_matrix.png assets/train_confusion_matrix.png assets/val_confusion_matrix.png
	
	

mshow:
	tensorboard --logdir multirun/ --load_fast=false --bind_all  &

sshow:
	tensorboard --logdir outputs/ --load_fast=false --bind_all &


showoff:
	# kill -9 $(lsof -ti :6006)
	@PID=$$(lsof -ti :6006); \
	if [ -n "$$PID" ]; then \
		echo "Killing process $$PID"; \
		/usr/bin/kill -9 $$PID; \
	else \
		echo "No process found on port 6006"; \
	fi
