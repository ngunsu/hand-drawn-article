import typer
import torch
import time
from torchinfo import summary
from point_detection.pl.point_detection_module import PointDetectionModule

app = typer.Typer()


def measure_forward_pass_time(model, input_data, warmup_iterations=50, timing_iterations=1000):
    device = torch.device('cuda')
    model.to(device).eval()  # Set the model to evaluation mode
    input_data = input_data.to(device)

    # Warmup iterations
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_data)
            torch.cuda.synchronize(device)  # Synchronize after each warmup iteration

    # Timing iterations
    start_time = time.time()
    with torch.no_grad():
        for _ in range(timing_iterations):
            _ = model(input_data)
            torch.cuda.synchronize(device)  # Synchronize after each timing iteration
    end_time = time.time()

    # Calculate and return the average time per iteration in milliseconds
    avg_time_ms = (end_time - start_time) / timing_iterations * 1000
    return avg_time_ms


@app.command()
def perform(checkpoint: str):
    """Eval"""
    cp = torch.load(checkpoint)
    hparams = cp['hyper_parameters']

    model = PointDetectionModule(hparams)
    model.load_state_dict(cp['state_dict'])

    summary(model.model, input_size=(1, 3, 128, 128))

    input = torch.randn((1, 3, 128, 128))
    avg_time_ms = measure_forward_pass_time(model.model, input)
    print(f"The average forward pass time over {1000} iterations is {avg_time_ms:.3f} milliseconds")


def main():
    app()


if __name__ == "__main__":
    main()
