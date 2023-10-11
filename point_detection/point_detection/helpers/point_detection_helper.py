import time
from point_detection.pl.point_detection_datamodule import FondefWorkpiecesPointDataModule
from point_detection.pl.point_detection_module import PointDetectionModule
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def gen_random_exp_id():
    exp_id = f'random_{time.time()}'
    return exp_id


def train_point_detector(params):

    seed_everything(params['seed'])

    # Model
    model = PointDetectionModule(params)

    # Datamodule
    dm = FondefWorkpiecesPointDataModule(split_type=params['split_type'],
                                         batch_size=params['batch_size'],
                                         im_size=params['im_size'],
                                         shuffle=params['shuffle'],
                                         num_workers=params['num_workers'])

    # Checkpoint callback
    callbacks = []
    exp_id = params['exp_id']
    if exp_id == "":
        exp_id = gen_random_exp_id()

    if not (params['just_test'] or params['just_val']):
        dirpath = f'{params["output_path"]}/{exp_id}/'
        checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                              save_top_k=params["save_top_k"],
                                              verbose=True,
                                              monitor='val_loss',
                                              mode='min')
        callbacks.append(checkpoint_callback)

        early_stopping = EarlyStopping('val_loss',
                                       min_delta=0.00,
                                       patience=params['patience'],
                                       verbose=False,
                                       check_finite=False,
                                       mode="min")
        callbacks.append(early_stopping)

    # Logger
    logger = None
    if not (params["just_test"] or params['just_val']):
        logger_path = f'{params["output_path"]}/{exp_id}/log'
        logger = TensorBoardLogger(logger_path)

    # Set trainer
    trainer = Trainer(devices=1,
                      precision=params['precision'],
                      accelerator='gpu',
                      deterministic=False,
                      accumulate_grad_batches=params['accumulate_grad_batches'],
                      logger=logger,
                      log_every_n_steps=params['log_every_n_steps'],
                      check_val_every_n_epoch=params['check_val_every_n_epoch'],
                      fast_dev_run=params['fast_dev_run'],
                      min_epochs=params['min_epochs'],
                      max_epochs=params['max_epochs'],
                      callbacks=callbacks)

    trainer.fit(model, datamodule=dm)
    return early_stopping.best_score.float().item()  # type: ignore
