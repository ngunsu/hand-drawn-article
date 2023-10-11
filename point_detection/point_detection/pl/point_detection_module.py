import torch
from point_detection.models.model_factory import model_factory
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pytorch_lightning as pl
from kornia import augmentation as K


class PointDetectionModule(pl.LightningModule):

    # -------------------------------------------------------------------
    # Training details - Network definition
    # -------------------------------------------------------------------
    def __init__(self, hparams: dict):
        """ Constructor

        Params
        ------
        hparams (dict): Contains the training configuration details
        """
        super().__init__()

        self.hparams.update(hparams)  # type: ignore
        self.model = model_factory(hparams)
        self.loss_fun = self.get_loss(hparams['loss'])
        self.configure_augmentation()

    # -------------------------------------------------------------------
    # Training details - Data augmentation
    # -------------------------------------------------------------------
    def configure_augmentation(self):
        self.train_augmentation = K.AugmentationSequential(K.Resize((self.hparams['im_size'], self.hparams['im_size'])),
                                                           K.RandomHorizontalFlip(p=0.5),
                                                           K.RandomVerticalFlip(p=0.5),
                                                           K.RandomAffine(90, p=0.2),
                                                           K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
                                                           K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                                                       std=torch.tensor([0.229, 0.224, 0.225])),
                                                           data_keys=['input', 'keypoints'],
                                                           same_on_batch=False)

        self.valtest_augmentation = torch.nn.Sequential(
            K.Resize((self.hparams['im_size'], self.hparams['im_size'])),
            # Normalize using ImageNet mean and standard deviation
            K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
        )

    # -------------------------------------------------------------------
    # Training details - Optimizer
    # -------------------------------------------------------------------
    def configure_optimizers(self):
        hparams = self.hparams

        # Optimizer
        if hparams['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=hparams['lr'], betas=(0.9, 0.999))
        elif hparams['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=hparams['lr'], momentum=0.9)
        elif hparams['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=hparams['lr'], betas=(0.9, 0.999))
        else:
            raise NameError(f'Optimizer {hparams["optimizer"]} not supported')

        # Scheduler
        if hparams['scheduler'] == 'steplr':
            scheduler = StepLR(optimizer, step_size=hparams['gamma_step'], gamma=hparams['gamma'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        elif hparams['scheduler'] == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=30, factor=hparams['gamma'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        else:
            return {'optimizer': optimizer}

    # -------------------------------------------------------------------
    # Training details - Loss
    # -------------------------------------------------------------------
    def get_loss(self, loss):
        if loss == 'mse':
            return torch.nn.MSELoss(reduction='sum')
        elif loss == 'smooth':
            return torch.nn.SmoothL1Loss(reduction='sum')
        else:
            exit(f'{loss} loss is not supported')

    # -------------------------------------------------------------------
    # Training details - Forward
    # -------------------------------------------------------------------
    def forward(self, x):
        return self.model.forward(x)

    # -------------------------------------------------------------------
    # Training details - Train step
    # -------------------------------------------------------------------
    def training_step(self, batch, _):
        ims, gt = batch

        # Fine tuning
        if self.hparams['freeze']:
            for p in self.parameters():
                p.requires_grad = False
            if self.hparams['model_type'] in ['efficientnet', 'convnexttiny', 'convnextlarge']:
                for p in self.model.classifier.parameters():  # type: ignore
                    p.requires_grad = True
            else:
                for p in self.model.fc.parameters():  # type: ignore
                    p.requires_grad = True
        # Augmentation
        with torch.no_grad():
            ims_aug, gt_aug = self.train_augmentation(ims, gt.unsqueeze(1) * self.hparams['im_size'])
            gt_aug = gt_aug.squeeze(1) / 128

        output = self.forward(ims_aug)
        loss = self.loss_fun(output, gt_aug)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # -------------------------------------------------------------------
    # Training details - Validation step
    # -------------------------------------------------------------------
    def validation_step(self, batch, _):
        ims, gt = batch

        ims = self.valtest_augmentation(ims)
        output = self.forward(ims)

        gt = gt * self.hparams['im_size']
        output = output * self.hparams['im_size']

        loss = torch.sqrt(torch.nn.functional.mse_loss(output, gt, reduction='sum'))  # type: ignore

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    # -------------------------------------------------------------------
    # Test details - Test step
    # -------------------------------------------------------------------
    def test_step(self, batch, _):
        ims, gt = batch

        ims = self.valtest_augmentation(ims)
        output = self.forward(ims)

        gt = gt * self.hparams['im_size']
        output = output * self.hparams['im_size']

        loss = torch.sqrt(torch.nn.functional.mse_loss(output, gt, reduction='sum'))  # type: ignore

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
