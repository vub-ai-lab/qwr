#!/bin/bash
#
# Usage: run_optuna <name> <env> <cores>

for i in $(seq $3)
do
	OMP_NUM_THREADS=1 ./optuna_tune.py ./main.py --name $1 --stdout R:3 --env $2 --max_episodes 200 --hidden 64 --layers 2 --lr 1e-3 --erfreq 10 --action_samples i:8:16 --gamma 0.98 --topk i:1:8 --beta f:1.0:10.0 &
	sleep 2
done

wait
