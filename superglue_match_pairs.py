from pathlib import Path
import argparse
import random
import os

import torch
import numpy as np
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors

from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import make_matching_plot, AverageTimer, read_image

from NetVLAD.dataset import get_whole_val_set

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and visualization with Utility-guided SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='',
        help='Path to the directory that contains the images')
    parser.add_argument('--dataset', type=str, default='berlin', 
        help='Dataset to use', choices=['oxford', 'nordland', 'berlin'])
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
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
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    parser.add_argument(
        '--k', type=int, default=10,
        help='Number of Top Utility Clusters')
    parser.add_argument(
        '--netvlad_extracts_path', type=str, default='',
        help='Path to NetVLAD Extractions')
    parser.add_argument(
        '--utility_path', type=str, default='',
        help='Path to Folder containing PS Utility and Low ES Utility Clusters')

    parser.add_argument(
        '--es_utility', action='store_true',
        help='Use Environment-Specific Utility')
    parser.add_argument(
        '--ps_utility', action='store_true',
        help='Use Place-Specific Utility')
    parser.add_argument(
        '--non_default_k', action='store_true',
        help='Use Non Default Number of Top Utility Clusters for Combined ES and PS Utility')
    
    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

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

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    matches_dir = Path(os.path.join(output_dir, 'matches'))
    matches_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(matches_dir))
    if opt.viz:
        viz_dir = Path(os.path.join(output_dir, 'viz'))
        viz_dir.mkdir(exist_ok=True, parents=True)
        print('Will write visualization images to',
              'directory \"{}\"'.format(viz_dir))
        
    # Loading cluster masks & netvlad candidates
    db_cluster_masks = np.load(os.path.join(opt.netvlad_extracts_path, 'db_cluster_masks.npy'))
    netvlad_preds = np.load(os.path.join(opt.netvlad_extracts_path, 'netvlad_preds.npy'))
    netvlad_candidates = netvlad_preds.flatten()

    # Load Reference Map Utilities
    ps_utility = np.load(os.path.join(opt.utility_path, 'ps_utility.npy'))
    ps_utility_rankings = np.flip(np.argsort(ps_utility, axis=1), axis=1)
    
    low_es_utility_clusters = np.load(os.path.join(opt.utility_path, 'low_es_utility_clusters.npy'))

    idx = np.in1d(ps_utility_rankings, low_es_utility_clusters)
    idx = ~idx
    combined_utility_rankings = ps_utility_rankings.flatten()[idx].reshape((ps_utility.shape[0], 
                                            (ps_utility.shape[1] - low_es_utility_clusters.shape[0])))

    if not opt.es_utility and not opt.ps_utility:
        topk = ps_utility_rankings                                                       # All Clusters
    elif opt.es_utility and not opt.ps_utility:
        topk = combined_utility_rankings                                                 # High ES Utility Clusters (X)
    elif opt.ps_utility and not opt.es_utility:
        topk = ps_utility_rankings[:, :(opt.k)]                                          # Top K PS Utility Clusters
    elif opt.ps_utility and opt.es_utility:
        if opt.non_default_k:
            topk = combined_utility_rankings[:, :(opt.k)]                                # Top K ES & PS Utility Clusters
        else:
            topk = combined_utility_rankings[:, :(combined_utility_rankings.shape[1]-1)] # Top X-1 ES & PS Utility Clusters

    SG_matches = []
    
    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = matches_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        if opt.viz:
            viz_path = viz_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)

        # Handle --cache logic.
        do_match = True
        do_viz = opt.viz
        if opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
                SG_matches.append(out_matches)
                do_match = False
            if opt.viz and viz_path.exists():
                do_viz = False
            timer.update('load_cache')

        if not (do_match or do_viz):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, opt.resize, rot1, opt.resize_float)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')
        
        # Getting Top K Utility Clusters and Cluster Masks for Reference Image
        db_cluster_mask = db_cluster_masks[netvlad_candidates[i].astype('int')]
        db_topk = topk[netvlad_candidates[i].astype('int')]

        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1}, db_cluster_mask, db_topk)
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            SG_matches.append(out_matches)
            np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        if do_viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'Utility-guided SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
                'Match Score: {}'.format(len(mkpts0)/(len(kpts0)+len(kpts1))),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    # Evaluation
    match_scores = np.zeros((netvlad_preds.shape[0], netvlad_preds.shape[1]))

    for i in range(match_scores.shape[0]):
        for j in range(match_scores.shape[1]):
            result = SG_matches[match_scores.shape[1]*i+j]
            total_keypoints = result['keypoints0'].shape[0] + result['keypoints1'].shape[0]
            inlier_count = np.sum(result['matches']>-1)
            match_scores[i,j] = inlier_count/total_keypoints

    dataset = get_whole_val_set(opt.input_dir, opt.dataset.lower())

    knn = NearestNeighbors(n_jobs=1)
    knn.fit(dataset.dbStruct.locDb)
    _ , gt = knn.radius_neighbors(dataset.dbStruct.locQ,
        radius=dataset.dbStruct.posDistThr)

    predictions = np.zeros((netvlad_preds.shape[0], netvlad_preds.shape[1]))

    for i in range(predictions.shape[0]):
        predictions[i,:] = netvlad_preds[i, np.flip(np.argsort(match_scores[i,:]))]

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    correct_at_n = np.zeros(len(n_values))

    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break

    recall_at_n = correct_at_n / dataset.dbStruct.numQ

    for i,n in enumerate(n_values):
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    np.save(os.path.join(output_dir, 'superglue_preds.npy'), predictions)
    np.savetxt(os.path.join(output_dir, 'superglue_recalls.txt'), recall_at_n)