# PYTHONPATH=. python3.11 src/embed.py --model_name_or_path contriever

# PYTHONPATH=.  python ric/main_ric.py --config-name example_config

BEAKER_REPLICA_RANK=1 BEAKER_REPLICA_COUNT=2 python -m ric.main_ric --config-name example_config

# python -m ric.main_ric --config-name example_config

# export PYTHONPATH=.

# torchrun \
#     --nnodes=1 \
#     --nproc-per-node=4 \
#     -m ric.main_ric --config-name example_config 


