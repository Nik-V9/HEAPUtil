import torch
import numpy as np

from .superpoint import SuperPoint
from .superglue import SuperGlue


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data, db_cluster_mask, db_topk):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
          db_cluster_mask: reference image cluster mask
          db_topk: Utility clusters to filter SuperPoint keypoints
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})

            # Filter SuperPoint using Utility
            db_keypoints = np.flip(pred1['keypoints'][0].detach().cpu().numpy(), axis=1)
            db_map = db_cluster_mask[db_keypoints.astype('int')[:,0].T,db_keypoints.astype('int')[:,1].T]
            db_filter = np.in1d(db_map, db_topk)
            db_filter_ind = np.where(db_filter == 1)[0]
            pred1['keypoints'] = [pred1['keypoints'][0][db_filter_ind,:]]
            pred1['scores'] = (pred1['scores'][0][db_filter_ind],)
            pred1['descriptors'] = [pred1['descriptors'][0][:, db_filter_ind]]

            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred
