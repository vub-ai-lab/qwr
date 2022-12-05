#!/usr/bin/python3
import optuna

import sys
import subprocess

# Usage: optuna_tune.py <command> [--name study_name] [--stdout filter:column] [--arg value] [--arg i:from:to] [--arg f:from:to] [--arg fl or il:from:to with log=True]
# For instance: optuna_tune.py main.py --stdout R:3 --erpoolsize i:100:10000
state = 'command'
command = ''
study_name = 'study'
stdout_key = 'R'
stdout_column = 3
key = ''
args = []

for cmdline_arg in sys.argv[1:]:
    if state == 'command':
        command = cmdline_arg
        state = 'key'

    elif state == 'key':
        if cmdline_arg == '--stdout':
            state = 'stdout_key'
        elif cmdline_arg == '--name':
            state = 'study_name'
        else:
            key = cmdline_arg
            state = 'value'

    elif state == 'stdout_key':
        stdout_key, stdout_column = cmdline_arg.split(':')
        stdout_column = int(stdout_column)
        state = 'key'

    elif state == 'study_name':
        study_name = cmdline_arg
        state = 'key'

    elif state == 'value':
        value = cmdline_arg

        if ':' not in value:
            args.append({
                'key': key,
                'value': value,
                'constant': True
            })
        else:
            datatype, fr, to = value.split(':')

            if datatype in ['i', 'il']:
                args.append({
                    'key': key,
                    'float': False,
                    'constant': False,
                    'from': int(fr),
                    'to': int(to),
                    'log': datatype == 'il'
                })
            elif datatype in ['f', 'fl']:
                args.append({
                    'key': key,
                    'float': True,
                    'constant': False,
                    'from': float(fr),
                    'to': float(to),
                    'log': datatype == 'fl'
                })

        state = 'key'

# Optuna objective: run the command line and observe its stdout
def objective(trial):
    process_args = []

    # Command
    process_args.append(command)

    # Arguments
    for arg in args:
        process_args.append(arg['key'])

        if arg['constant']:
            value = arg['value']
        elif arg['float']:
            value = trial.suggest_float(arg['key'][2:], arg['from'], arg['to'], log=arg['log'])
        else:
            value = trial.suggest_int(arg['key'][2:], arg['from'], arg['to'], log=arg['log'])

        process_args.append(str(value))

    # Start the process
    print('PROCESS', process_args)
    p = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding='utf8')
    value = None
    episode_index = 0

    for line in p.stdout:
        if not line.startswith(stdout_key):
            continue

        parts = line.strip().split()
        v = float(parts[stdout_column])
        episode_index += 1

        if value is None:
            value = v
        else:
            value = 0.9 * value + 0.1 * v

        if episode_index > 50:
            trial.report(value, episode_index)

            if trial.should_prune():
                p.terminate()
                raise optuna.TrialPruned()

    return value

study = optuna.create_study(
    storage="sqlite:///optuna.sqlite",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    direction='maximize',
    study_name=study_name,
    load_if_exists=True
)
study.optimize(objective, n_trials=10000)
