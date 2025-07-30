# ====== Config ======
IMAGE        ?= rolling-bair:cu121
TFDS_DIR     ?= $(HOME)/tensorflow_datasets
APP_DIR      ?= $(PWD)

# Training params (override on CLI as needed)
FRAMES       ?= 16
COND_FRAMES  ?= 1
BATCH        ?= 8
BASE         ?= 64
STEPS        ?= 1000
CKPT_EVERY   ?= 100

# Sampling params
SPLIT        ?= test
REFINE_PASSES?= 2
SAMPLE_OUT   ?= samples/bair_long_final_second
CKPT         ?= checkpoints/ema_final.pt

# Docker run common flags
RUN_GPU      ?= --gpus all
MOUNTS       ?= -v $(APP_DIR):/app -v $(TFDS_DIR):/data/tfds
ENVVARS      ?= -e TFDS_DATA_DIR=/data/tfds -w /app


.PHONY: tfds
TFDS_NAME ?= bair_robot_pushing_small
tfds:
	@mkdir -p "$(TFDS_DIR)"
	docker run --rm -v "$(TFDS_DIR)":/data/tfds python:3.10-slim \
	bash -lc "pip install -q tensorflow-datasets && \
	          python -c \"import tensorflow_datasets as tfds; \
	                      tfds.load('$(TFDS_NAME)', data_dir='/data/tfds', split='train', download=True); \
	                      print('Done')\""


.PHONY : build
build:
	docker build -t $(IMAGE) .

.PHONY: train
train: build
	docker run --rm -it $(RUN_GPU) \
		$(MOUNTS) $(ENVVARS) $(IMAGE) \
		uv run python train_bair_rolling.py \
			--data-dir "$$TFDS_DATA_DIR" \
			--save-dir "checkpoints" \
			--frames $(FRAMES) \
			--cond-frames $(COND_FRAMES) \
			--mode rolling \
			--batch $(BATCH) \
			--base $(BASE) \
			--fp16 \
			--steps $(STEPS) \
			--ckpt-every $(CKPT_EVERY)

.PHONY: sample
sample: build
	docker run --rm -it $(RUN_GPU) \
		$(MOUNTS) $(ENVVARS) $(IMAGE) \
		uv run python sample_bair_rolling.py \
			--ckpt "$(CKPT)" \
			--out "$(SAMPLE_OUT)" \
			--frames $(FRAMES) \
			--cond-frames $(COND_FRAMES) \
			--base $(BASE) \
			--batch 4 \
			--split $(SPLIT) \
			--refine-passes $(REFINE_PASSES)