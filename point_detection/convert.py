import typer
import torch
from point_detection.pl.point_detection_module import PointDetectionModule

app = typer.Typer()


@app.command()
def convert(checkpoint: str):
    """Eval"""
    cp = torch.load(checkpoint)
    hparams = cp['hyper_parameters']

    model = PointDetectionModule(hparams)
    model.load_state_dict(cp['state_dict'])

    input_sample = torch.randn((1, 3, 128, 128))
    model.to_onnx('out.onnx', input_sample)


def main():
    app()


if __name__ == "__main__":
    main()
