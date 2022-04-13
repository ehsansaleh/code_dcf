.PHONY: summary venv tables filelists

SHELL := /bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
MPIPRECMD := $(shell command -v mpirun >/dev/null 2>&1 && echo "mpirun -n 10")

##########################################################
####################      VENV     #######################
##########################################################
venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip
	source venv/bin/activate && pip install torch==1.7.1+cu101 \
		torchvision==0.8.2+cu101 \
		-f https://download.pytorch.org/whl/torch_stable.html
	source venv/bin/activate && python -m pip install -r requirements.txt

##########################################################
###########      Summarizing CSV Results     #############
##########################################################
summary:
	source .env.sh && cd utils && ${MPIPRECMD} python csv2summ.py

tables:
	source .env.sh && cd utils && python summ2tbls.py

##########################################################
####################   Filelists   #######################
##########################################################
filelists:
	source .env.sh && cd filelists &&  python json_maker.py

##########################################################
#####################     Clean     ######################
##########################################################
fix_crlf:
	find ${PROJBASE} -maxdepth 3 -type f -name "*.md5" \
	  -o -name "*.py" -o -name "*.sh" -o -name "*.json" | xargs dos2unix

clean:
	@echo Nothing to clean
