#!/bin/bash
#
# Usage: run_optuna <name> <env> <cores>

for i in $(seq $3)
do
	OMP_NUM_THREADS=1 ./optuna_tune.py ./main.py --name $1 --stdout R:3 --env $2 --max_episodes 200 --hidden il:32:512 --layers i:1:3 --lr fl:1e-5:1e-2 --erfreq il:10:1000 --action_samples il:4:32 --gamma 0.98 --tau fl:1.0:10.0 --beta fl:1.0:10.0 &
	sleep 2
done

wait
