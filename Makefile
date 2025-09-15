# Simple Makefile helpers
.DEFAULT_GOAL := help

PY ?= python
DATA_PATH ?= ./example_data
FILE_EXT ?= npy
METHODS ?= simclr moco dino2 dino3

# Common training knobs
CHANNELS ?= 15
EPOCHS ?= 50
BS ?= 32
LR ?= 1e-3

# Algorithm-specific (can override)
MOCO_LR ?= 6e-2

# Pretrained control
PRETRAINED_PATH ?=
NO_TIMM ?= 1
TIMM_FLAG := $(if $(NO_TIMM),--no_timm_pretrained,)
PRETRAINED_FLAG := $(if $(PRETRAINED_PATH),--pretrained_path $(PRETRAINED_PATH),)

.PHONY: help smoke install train-simclr train-moco train-dino2 train-dino3 train-finetune-binary train-finetune-multi

help:
	@echo "Available targets:"
	@echo "  install     Install Python dependencies (requirements.txt)"
	@echo "  smoke       Run smoke_test.py on a tiny batch"
	@echo "  train-simclr  Run SimCLR training (train_ssl.py)"
	@echo "  train-moco    Run MoCo v3 training (train_moco_v3.py)"
	@echo "  train-dino2   Run DINOv2 training (train_dino_v2.py)"
	@echo "  train-dino3   Run DINOv3 training (train_dino_v3.py)"
	@echo "  train-finetune-binary   Fine-tune classifier (binary) on TIFFs"
	@echo "  train-finetune-multi    Fine-tune classifier (multiclass) on TIFFs"
	@echo "Variables: DATA_PATH=./example_data FILE_EXT=npy METHODS=\"simclr moco dino2 dino3\""

install:
	$(PY) -m pip install -r requirements.txt

smoke:
	MPLCONFIGDIR=.cache $(PY) smoke_test.py --data_path $(DATA_PATH) --file_extension $(FILE_EXT) --methods $(METHODS)

train-simclr:
	$(PY) train_ssl.py --data_path $(DATA_PATH) --file_extension $(FILE_EXT) \
	  --num_channels $(CHANNELS) --epochs $(EPOCHS) --batch_size $(BS) --lr $(LR) \
	  $(PRETRAINED_FLAG) $(TIMM_FLAG)

train-moco:
	$(PY) train_moco_v3.py --data_path $(DATA_PATH) --file_extension $(FILE_EXT) \
	  --num_channels $(CHANNELS) --epochs $(EPOCHS) --batch_size $(BS) --lr $(MOCO_LR) \
	  $(PRETRAINED_FLAG) $(TIMM_FLAG)

train-dino2:
	$(PY) train_dino_v2.py --data_path $(DATA_PATH) --file_extension $(FILE_EXT) \
	  --num_channels $(CHANNELS) --epochs $(EPOCHS) --batch_size $(BS) --lr $(LR) \
	  $(PRETRAINED_FLAG) $(TIMM_FLAG)

train-dino3:
	$(PY) train_dino_v3.py --data_path $(DATA_PATH) --file_extension $(FILE_EXT) \
	  --num_channels $(CHANNELS) --epochs $(EPOCHS) --batch_size $(BS) --lr $(LR) \
	  $(PRETRAINED_FLAG) $(TIMM_FLAG)

train-finetune-binary:
	$(PY) train_finetune.py --data_path $(DATA_PATH) --task binary --num_channels $(CHANNELS) \
	  --epochs $(EPOCHS) --batch_size $(BS) --lr $(LR) $(PRETRAINED_FLAG) $(TIMM_FLAG)

train-finetune-multi:
	$(PY) train_finetune.py --data_path $(DATA_PATH) --task multiclass --num_channels $(CHANNELS) \
	  --epochs $(EPOCHS) --batch_size $(BS) --lr $(LR) $(PRETRAINED_FLAG) $(TIMM_FLAG)
