from pathlib import Path
import argparse
import random
import os

import torch
import numpy as np

from models.superpoint import SuperPoint
from models.utils import AverageTimer, read_image

torch.set_grad_enabled(False)

class descriptor_extraction(torch.nn.Module):
    """ Image Descriptor Frontend (SuperPoint) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))

    def forward(self, data):
        """ Run SuperPoint
        Args:
          data: dictionary with minimal keys: ['image0']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        pred0 = self.superpoint({'image': data['image0']})

        return pred0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract SuperPoint features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_images', type=str, default='',
        help='Path to the list of images (text file)')
    parser.add_argument(
        '--input_dir', type=str, default='',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--split', type=str, default='db',
        help='Reference (db) or Query', choices=['db', 'query'])
    parser.add_argument(
        '--output_dir', type=str, default='',
        help='Path to save SuperPoint Extractions')
    parser.add_argument(
        '--max_keypoints', type=int, default=2048,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=3,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    
    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.input_images, 'r') as f:
        images = f.read().splitlines()

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        }
    }
    
    sp = descriptor_extraction(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write Super Point descriptors to directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)

    lst = []
    
    for i in range(len(images)):
        name0 = images[i]
        stem0 = Path(name0).stem
        # If a rotation integer is provided (e.g. from EXIF data), use it:
        rot0 = 0

        # Load the image.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, opt.resize, rot0, opt.resize_float)
        if image0 is None:
            print('Problem reading image: {}'.format(
                input_dir/name0))
            exit(1)
        timer.update('load_image')      

        # SuperPoint descriptors extraction
        pred = sp({'image0': inp0})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts = pred['keypoints']
        scores = pred['scores']
        descriptors = pred['descriptors']
        timer.update('matcher')
        out_desc = {'keypoints': kpts, 'descriptors': descriptors,
                       'scores': scores}
        lst.append(out_desc)
            
        timer.print('Finished image {:5} of {:5}'.format(i, len(images)))
            
    np.savez(os.path.join(output_dir, opt.split+'.npz'), lst)