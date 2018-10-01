#!/bin/bash
set -e

DATASET="ml-20m"

BUCKET=${BUCKET:-""}
ROOT_DIR="${BUCKET:-/tmp}/MLPerf_NCF"
echo "Root directory: ${ROOT_DIR}"

if [[ -z ${BUCKET} ]]; then
  LOCAL_ROOT=${ROOT_DIR}
else
  LOCAL_ROOT="/tmp/MLPerf_NCF"
  mkdir -p ${LOCAL_ROOT}
  echo "Local root (for files which cannot use GCS): ${LOCAL_ROOT}"
fi

DATE=$(date '+%Y-%m-%d_%H:%M:%S')
TEST_DIR="${ROOT_DIR}/${DATE}"
LOCAL_TEST_DIR="${LOCAL_ROOT}/${DATE}"
mkdir -p ${LOCAL_TEST_DIR}

TPU=${TPU:-""}
if [[ -z ${TPU} ]]; then
  DEVICE_FLAG="--num_gpus -1"
else
  DEVICE_FLAG="--tpu ${TPU} --num_gpus 0"
fi

DATA_DIR="${ROOT_DIR}/movielens_data"
python ../datasets/movielens.py --data_dir ${DATA_DIR} --dataset ${DATASET}

{

for i in `seq 0 4`;
do
  START_TIME=$(date +%s)
  MODEL_DIR="${TEST_DIR}/model_dir_${i}"

  RUN_LOG="${LOCAL_TEST_DIR}/run_${i}.log"
  echo ""
  echo "Beginning run ${i}"
  echo "  Complete logs are in ${RUN_LOG}"

  # To reduce variation set the seed flag:
  #   --seed ${i}
  #
  # And to confirm that the pipeline is deterministic pass the flag:
  #   --hash_pipeline
  #
  # (`--hash_pipeline` will slow down training)
  python ncf_main.py --model_dir ${MODEL_DIR} \
                     --data_dir ${DATA_DIR} \
                     --dataset ${DATASET} --hooks "" \
                     ${DEVICE_FLAG} \
                     --clean \
                     --train_epochs 20 \
                     --batch_size 2048 \
                     --eval_batch_size 65536 \
                     --learning_rate 0.0005 \
                     --layers 256,256,128,64 --num_factors 64 \
                     --hr_threshold 0.635 \
                     --ml_perf \
  |& tee ${RUN_LOG} \
  | grep --line-buffered  -E --regexp="(Iteration [0-9]+: HR = [0-9\.]+, NDCG = [0-9\.]+)|(pipeline_hash)"

  END_TIME=$(date +%s)
  echo "Run ${i} complete: $(( $END_TIME - $START_TIME )) seconds."

done

} |& tee "${LOCAL_TEST_DIR}/summary.log"
