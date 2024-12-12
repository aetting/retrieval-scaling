# # Generate a hash for the run name by combining model name and documents
# RUN_HASH=$(echo -n "${MODEL_NAME}${DOCUMENTS}" | md5sum | awk '{print $1}')
# RUN_NAME="nvidia_deberta_${RUN_HASH:0:8}"

# # Set the run name as an environment variable
# export BEAKER_EXPERIMENT_NAME="${RUN_NAME}"

CLUSTER="ai2/mosaic*"
PRIORITY="high"

export BEAKER_EXPERIMENT_NAME="Contriever-mergedqa-prefilter-sample"

gantry run \
    --task-name "Contriever-mergedqa-prefilter-sample" \
    --description "Embed docs for dense retrieval" \
    --allow-dirty \
    --workspace ai2/reddit \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 1 \
    --replicas 1 \
    --preemptible \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --shared-memory 10GiB \
    --weka oe-data-default:/data \
    --yes \
    -- python -m ric.main_ric --config-name example_config tasks.datastore.index=true
