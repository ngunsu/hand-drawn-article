import typer
import torch
import os
import torchvision
from pytorch_lightning.trainer import Trainer
from point_detection.pl.point_detection_module import PointDetectionModule
from point_detection.pl.point_detection_datamodule import FondefWorkpiecesPointDataModule

app = typer.Typer()


@app.command()
def eval(checkpoint: str, save: bool = False):
    """Eval"""
    cp = torch.load(checkpoint)
    hparams = cp['hyper_parameters']

    model = PointDetectionModule(hparams)
    model.load_state_dict(cp['state_dict'])

    # Datamodule
    dm = FondefWorkpiecesPointDataModule(split_type=hparams['split_type'],
                                         batch_size=hparams['batch_size'],
                                         im_size=hparams['im_size'],
                                         shuffle=hparams['shuffle'],
                                         num_workers=hparams['num_workers'])
    trainer = Trainer(devices=1,
                      precision=hparams['precision'],
                      accelerator='gpu',
                      deterministic=False,
                      accumulate_grad_batches=hparams['accumulate_grad_batches'])
    trainer.test(model, dm)

    if save:
        os.system('mkdir -p ./results')
        model.eval()
        with torch.no_grad():
            dm_test = dm.test_dataloader()
            image_with_keypoints = []
            for x, y in dm_test:
                out = model(model.valtest_augmentation(x))
                gt = torch.tensor(y.detach().unsqueeze(0)*128, dtype=torch.uint8)
                out = torch.tensor(out.detach().unsqueeze(0)*128, dtype=torch.uint8)
                x = x.squeeze(0)
                k_im = torchvision.utils.draw_keypoints(torch.tensor(x.clone().detach()*255, dtype=torch.uint8),
                                                        gt, colors='red', radius=5)
                k_im = torchvision.utils.draw_keypoints(torch.tensor(k_im.clone().detach()*255, dtype=torch.uint8),
                                                        out, colors='yellow', radius=5)
                image_with_keypoints.append(k_im)
            grid = torchvision.utils.make_grid(image_with_keypoints)
            torchvision.io.write_png(grid, f'./results/{hparams["exp_id"]}.png')


def main():
    app()


if __name__ == "__main__":
    main()
