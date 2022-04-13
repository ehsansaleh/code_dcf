#!/bin/bash

PROJPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source .env.sh
cd $PROJPATH

################################################################################
###################### Checking the existence of filelists #####################
################################################################################
REQDS=("miniImagenet" "tieredImagenet" "CUB")
REQSPLITS=("base" "val" "novel")

MISSINGFSJSON=""
for DS in "${REQDS[@]}"; do
  for SPLITNAME in "${REQSPLITS[@]}"; do
    if [[ ! -f "./filelists/${DS}/${SPLITNAME}.json" ]]; then
      MISSINGFSJSON="./filelists/${DS}/${SPLITNAME}.json"
    fi
  done
done

if [[ ${MISSINGFSJSON} != "" ]]; then
  echo "Filelist ${MISSINGFSJSON} is missing. Running make filelists"
  make filelists || exit 0
fi

################################################################################
######################### Running the configs in a loop ########################
################################################################################
MODEL="WideResNet28_10"
METHOD="S2M2_R"
SOURCE_TARGET_SPLIT=("miniImagenet"    "miniImagenet"    "base"  \
                     "miniImagenet"    "miniImagenet"    "val"   \
                     "miniImagenet"    "miniImagenet"    "novel" \
                     "tieredImagenet"  "tieredImagenet"  "base"  \
                     "tieredImagenet"  "tieredImagenet"  "val"   \
                     "tieredImagenet"  "tieredImagenet"  "novel" \
                     "CUB"             "CUB"             "base"  \
                     "CUB"             "CUB"             "val"   \
                     "CUB"             "CUB"             "novel" \
                     "miniImagenet"    "CUB"             "val"   \
                     "tieredImagenet"  "CUB"             "val"   \
                     "miniImagenet"    "CUB"             "novel" \
                     "tieredImagenet"  "CUB"             "novel" )

for (( i=0; i<${#SOURCE_TARGET_SPLIT[*]}; i+=3)); do
  SOURCEDS=${SOURCE_TARGET_SPLIT[$i]}
  TARGETDS=${SOURCE_TARGET_SPLIT[$((i+1))]}
  SPLIT=${SOURCE_TARGET_SPLIT[$((i+2))]}
  echo "Running configuration $((i/3))"
  echo "  ==> Source Dataset == ${SOURCEDS}"
  echo "  ==> Target Dataset == ${TARGETDS}"
  echo "  ==> Target Split == ${SPLIT}"
  echo "  + python save_features.py --source_dataset ${SOURCEDS} --target_dataset ${TARGETDS} \\"
  echo "           --split ${SPLIT} --model ${MODEL} --method ${METHOD} --device cuda:0"
  python save_features.py --source_dataset ${SOURCEDS} --target_dataset ${TARGETDS} \
        --split ${SPLIT} --model ${MODEL} --method ${METHOD} --device cuda:0 
  echo "----------------------------------------"
done


