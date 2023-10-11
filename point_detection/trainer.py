import typer
from point_detection.helpers.point_detection_helper import train_point_detector
app = typer.Typer()


@app.command()
def train(exp_id: str = None,
          just_test: bool = False,
          just_val: bool = False,
          fast_dev_run: bool = False,
          save_results: bool = False,
          checkpoint: str = None,
          pretrained_model: bool = True,
          model_type: str = 'shufflenet',
          save_top_k: int = 1,
          precision: str = '32',
          min_epochs: int = 10,
          max_epochs: int = 50,
          patience: int = 10,
          log_every_n_steps: int = 10,
          check_val_every_n_epoch: int = 5,
          accumulate_grad_batches: int = 1,
          split_type: str = 'k4_1',
          batch_size: int = 64,
          im_size: int = 128,
          shuffle: bool = True,
          optimizer: str = 'adam',
          scheduler: str = 'step_lr',
          lr: float = 0.005,
          gamma_step: int = 10,
          gamma: float = 0.1,
          num_workers: int = 20,
          seed: int = 10,
          loss: str = 'smooth',
          freeze: bool = False,
          output_path: str = './output',
          save: bool = False,
          to_onnx: bool = False):
    """Trainer"""
    hparams = locals()
    val = train_point_detector(hparams)
    print(f'val_loss: {val}')


def main():
    app()


if __name__ == "__main__":
    main()
