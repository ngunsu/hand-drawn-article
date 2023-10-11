import os
import math
import numpy as np
import sys
import optuna

sys.path.append('/workspace')
from point_detection.helpers.point_detection_helper import train_point_detector

OPT_NAME = 'opt_efficientnet'


def objective(trial):
    """Optuna objective function.

    Parameters
    ----------
    trial :
        Optuna trial
    """
    os.system(f'rm -rf ./log/{OPT_NAME}')
    hparams = {}
    hparams['exp_id'] = ''
    hparams['just_test'] = False
    hparams['just_val'] = False
    hparams['fast_dev_run'] = False
    hparams['save_results'] = False
    hparams['checkpoint'] = None
    hparams['pretrained_model'] = True
    hparams['model_type'] = 'efficientnet'
    hparams['save_top_k'] = 1
    hparams['precision'] = '16-mixed'
    hparams['min_epochs'] = 10
    hparams['max_epochs'] = 200
    hparams['patience'] = 20
    hparams['log_every_n_steps'] = 10
    hparams['check_val_every_n_epoch'] = 1
    hparams['accumulate_grad_batches'] = 1
    hparams['split_type'] = 'k4_1'
    hparams['pixel_delta'] = 20
    hparams['batch_size'] = trial.suggest_int("batch_size", 32, 128, step=32)
    hparams['im_size'] = 128
    hparams['shuffle'] = True
    hparams['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw'])
    hparams['scheduler'] = trial.suggest_categorical('scheduler', ['steplr', 'plateau'])
    hparams['lr'] = trial.suggest_categorical('lr', [0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005])
    hparams['gamma_step'] = trial.suggest_int('gamma_step', 10, 50, step=5)
    hparams['gamma'] = trial.suggest_float('gamma', 0.05, 0.5, step=0.05)
    hparams['num_workers'] = 20
    hparams['seed'] = 10
    hparams['loss'] = 'smooth'
    hparams['freeze'] = trial.suggest_categorical('freeze', [True, False])
    hparams['output_path'] = f'./output_{OPT_NAME}'
    hparams['save'] = False
    k_fold = np.zeros(4)
    for i in range(4):
        hparams['split_type'] = f'k4_{i+1}'
        val = train_point_detector(hparams)
        if val is not None and not math.isnan(val):
            k_fold[i] = val
        else:
            k_fold[i] = 1000
    return k_fold.mean()


def run():
    study = optuna.create_study(study_name=OPT_NAME,  # type: ignore
                                direction='minimize',
                                storage=f'sqlite:///{OPT_NAME}.db',
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    run()
