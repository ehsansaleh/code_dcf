#!/bin/bash

PROJPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source .env.sh
cd $PROJPATH

################################################################################
######################## Checking if the features exist ########################
################################################################################

REQFEATS=("WideResNet28_10_S2M2_R/mini2mini_base" \
          "WideResNet28_10_S2M2_R/mini2CUB_novel" \
          "WideResNet28_10_S2M2_R/mini2mini_novel" \
          "WideResNet28_10_S2M2_R/tiered2tiered_base" \
          "WideResNet28_10_S2M2_R/tiered2CUB_novel" \
          "WideResNet28_10_S2M2_R/tiered2tiered_novel")
               
MISSINGFEAT=""
for FEATFILE in "${REQFEATS[@]}"; do
  if [[ ! -f "./features/${FEATFILE}.pkl" ]]; then
    MISSINGFEAT="./features/${FEATFILE}.pkl"
  fi
done

if [[ ${MISSINGFEAT} != "" ]]; then
  echo "Features file ${MISSINGFEAT} is missing. Running ./features/download.sh ..."
  ./features/download.sh
fi

################################################################################
######################### Running the configs in a loop ########################
################################################################################

CFGPREFIXLIST=("1_mini2CUB/1s10w_0aug" \
               "1_mini2CUB/1s15w_0aug" \
               "1_mini2CUB/5s10w_0aug" \
               "1_mini2CUB/5s15w_0aug" \
               "2_tiered2CUB/1s10w_0aug" \
               "2_tiered2CUB/1s15w_0aug" \
               "2_tiered2CUB/5s10w_0aug" \
               "2_tiered2CUB/5s15w_0aug" \
               "3_tiered2tiered/1s10w_0aug" \
               "3_tiered2tiered/1s15w_0aug" \
               "3_tiered2tiered/5s10w_0aug" \
               "3_tiered2tiered/5s15w_0aug" \
               "1_mini2CUB/1s10w_750aug" \
               "1_mini2CUB/1s15w_750aug" \
               "1_mini2CUB/5s10w_750aug" \
               "1_mini2CUB/5s15w_750aug" \
               "2_tiered2CUB/1s10w_750aug" \
               "2_tiered2CUB/1s15w_750aug" \
               "2_tiered2CUB/5s10w_750aug" \
               "2_tiered2CUB/5s15w_750aug" \
               "3_tiered2tiered/1s10w_750aug" \
               "3_tiered2tiered/1s15w_750aug" \
               "3_tiered2tiered/5s10w_750aug" \
               "3_tiered2tiered/5s15w_750aug" \
               "4_5ways/mini2mini_1s5w_750aug" \
               "4_5ways/mini2mini_5s5w_750aug" \
               "4_5ways/tiered2tiered_1s5w_750aug" \
               "4_5ways/tiered2tiered_5s5w_750aug")
               
mkdir -p joblogs
for CFGPREFIX in "${CFGPREFIXLIST[@]}"; do
  OUTLOG="./joblogs/$(echo ${CFGPREFIX} | tr / _).out"
  echo "Running Configuration $CFGPREFIX"
  echo "  ==> The json config will be read from ./configs/${CFGPREFIX}.json"
  echo "  ==> The results csv file will be saved at ./results/${CFGPREFIX}.csv"
  echo "  ==> The training logs will be saved at ${OUTLOG}"
  echo "  + python main.py --device cuda:0 --configid ${CFGPREFIX} > $OUTLOG 2>&1"
  python main.py --device cuda:0 --configid ${CFGPREFIX} > $OUTLOG 2>&1
  echo "----------------------------------------"
done

