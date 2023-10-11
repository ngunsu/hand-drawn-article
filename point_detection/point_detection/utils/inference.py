import numpy as np
import cv2
import torch
from torchvision import transforms
from point_detection.utils.triton_helper import TritonHelper


class PointLocalizator():

    precision = 'FP32'

    def __init__(self, url='localhost:8000', model_name='holes_best_shufflenet', im_size=(3, 128, 128), family='holes') -> None:
        """
        Constructor


        Parameters
        ----------
        url : Triton URL
        im_size : Image size used by model (width, height). Downsample resolution
        family: Type of detection
        """
        self.triton_helper = TritonHelper(url)
        self.inputs = None
        self.outputs = None
        self.family = family
        self.im_size = im_size
        self.model_name = model_name
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((im_size[2], im_size[1]))])

    def preprocessing(self, im) -> np.ndarray:
        im_transformed = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # type: ignore
        im_transformed = self.transform(im_transformed).detach().numpy()
        return im_transformed

    def postprocessing(self, x, im_in) -> dict:
        return x  # type: ignore

    def get_in_out(self):
        inputs = [{'name': 'input.1', 'shape': self.im_size, 'precision': self.precision}]
        outputs = [{'name': '1152'}]
        return inputs, outputs

    def localize(self, im: np.ndarray):  # type: ignore
        if self.inputs is None:
            inputs, outputs = self.get_in_out()  # type: ignore
            self.triton_helper.set_in_out(inputs, outputs)
        im_p = self.preprocessing(im)
        response = self.triton_helper.call_server(im_p, 'input.1', self.model_name, '1', '1')  # type: ignore
        output = response.as_numpy('1152')
        return self.postprocessing(output, im)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Image path', required=True)
    args = parser.parse_args()

    pl = PointLocalizator()
    im = cv2.imread(args.image)   # type: ignore
    pt = pl.localize(im)[0]
    pt[0] *= im.shape[1]
    pt[1] *= im.shape[0]
    cv2.circle(im, pt.astype(int), 5, (255, 5, 120), 2)  # type: ignore
    cv2.imwrite('out.png', im)  # type: ignore
