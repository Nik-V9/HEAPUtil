from __future__ import print_function
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import io

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

import imageio
import skimage.io
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color

from NetVLAD.dataset import get_whole_val_set

parser = argparse.ArgumentParser(description='Utility-guided Local Feature Matching')
parser.add_argument(
    '--input_dir', type=str, default='',
    help='Path to the directory that contains the dataset')
parser.add_argument('--dataset', type=str, default='berlin', 
    help='Dataset to use', choices=['oxford', 'nordland', 'berlin'])
parser.add_argument(
    '--output_dir', type=str, default='',
    help='Path to the output directory to which the results and optionally, the visualizations are saved')
parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the best matches along with utility and dump as gif')
parser.add_argument(
    '--netvlad_extracts_path', type=str, default='',
    help='Path to NetVLAD Extractions')
parser.add_argument(
    '--superpoint_extracts_path', type=str, default='',
    help='Path to SuperPoint Extractions')
parser.add_argument(
    '--utility_path', type=str, default='',
    help='Path to Folder containing PS Utility and Low ES Utility Clusters')
parser.add_argument(
    '--k', type=int, default=10,
    help='Number of Top Utility Clusters')
parser.add_argument(
    '--es_utility', action='store_true',
    help='Use Environment-Specific Utility')
parser.add_argument(
    '--ps_utility', action='store_true',
    help='Use Place-Specific Utility')
parser.add_argument(
    '--non_default_k', action='store_true',
    help='Use Non Default Number of Top Utility Clusters for Combined ES and PS Utility')

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    return img

def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()

    return H, inliers

def drawMatches(imageA, imageB, kpsA, kpsB, matches):
	# initialize the output visualization image
	(hA, wA) = imageA.shape[:2]
	(hB, wB) = imageB.shape[:2]
	vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
	vis[0:hA, 0:wA] = imageA
	vis[0:hB, wA:] = imageB[:,:,:3]

	for i in range(len(kpsA)):
		ptA = (int(kpsA[i][1]), int(kpsA[i][0]))
		cv2.circle(vis, ptA, radius=3, color=(235, 204, 255), thickness=1)
	for i in range(len(kpsB)):
		ptB = (int(kpsB[i][1]) + wA, int(kpsB[i][0]))
		cv2.circle(vis, ptB, radius=3, color=(235, 204, 255), thickness=1)

	queryIdx = np.array([m.queryIdx for m in matches])
	trainIdx = np.array([m.trainIdx for m in matches])
	# loop over the matches
	for i in range(queryIdx.shape[0]):
		# draw the match
		ptA = (int(kpsA[queryIdx[i]][1]), int(kpsA[queryIdx[i]][0]))
		ptB = (int(kpsB[trainIdx[i]][1]) + wA, int(kpsB[trainIdx[i]][0]))

		cv2.circle(vis, ptA, radius=3, color=(255, 0, 0), thickness=1)
		cv2.circle(vis, ptB, radius=3, color=(255, 0, 0), thickness=1)
		cv2.line(vis, ptA, ptB, (0, 255, 0), 2, cv2.LINE_AA)

	return vis

if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)

    # Create the output directories if they do not exist already.
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Loading cluster masks & netvlad candidates
    db_cluster_masks = np.load(os.path.join(opt.netvlad_extracts_path, 'db_cluster_masks.npy'))
    netvlad_preds = np.load(os.path.join(opt.netvlad_extracts_path, 'netvlad_preds.npy'))

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

    # Load SuperPoint Extractions
    SP_all_db = np.load(os.path.join(opt.superpoint_extracts_path, 'db.npz'), allow_pickle=True)['arr_0']
    SP_all_query = np.load(os.path.join(opt.superpoint_extracts_path, 'query.npz'), allow_pickle=True)['arr_0']

    # Local Feature Matching
    LFM_matches = []
    match_scores = np.zeros((netvlad_preds.shape[0], netvlad_preds.shape[1]))

    for i in tqdm(range(0,match_scores.shape[0]), desc='LFM'):

        SP_q = SP_all_query[i]

        for j in range(0,match_scores.shape[1]):

            SP_db = SP_all_db[netvlad_preds[i, j].astype('int')]
            db_cluster_mask = db_cluster_masks[netvlad_preds[i, j].astype('int')]
            db_topk = topk[netvlad_preds[i, j].astype('int')]

            kp_q = SP_q['keypoints'].tolist()
            desc1 = SP_q['descriptors'].T

            db_keypoints = np.flip(SP_db['keypoints'], axis=1)
            db_map = db_cluster_mask[db_keypoints.astype('int')[:, 0].T, db_keypoints.astype('int')[:, 1].T]
            db_filter = np.in1d(db_map, db_topk)
            db_filter_ind = np.where(db_filter == 1)[0]

            kp_db = SP_db['keypoints'][db_filter_ind, :].tolist()
            desc2 = SP_db['descriptors'][:, db_filter_ind].T

            kp1 = [cv2.KeyPoint(p[1], p[0], 1) for p in kp_q]

            kp2 = [cv2.KeyPoint(p[1], p[0], 1) for p in kp_db]

            if len(kp1) != 0 and len(kp2) != 0:
                # Match and get rid of outliers
                m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
                H, inliers = compute_homography(m_kp1, m_kp2)

                # Draw SuperPoint matches
                matches = np.array(matches)[inliers.astype(bool)].tolist()

                out_matches = {'keypoints0': np.flip(SP_q['keypoints'], axis=1).tolist(),
                           'keypoints1': np.flip(SP_db['keypoints'][db_filter_ind, :], axis=1).tolist(),
                           'matches': matches}

                n_inlier = len(matches)
                match_scores[i, j] = n_inlier/(len(kp_q) + len(kp_db))
            else:
                out_matches = {'keypoints0': np.flip(SP_q['keypoints'], axis=1).tolist(), 
                            'keypoints1': np.flip(SP_db['keypoints'][db_filter_ind, :], axis=1).tolist()}
                match_scores[i, j] = 0

            LFM_matches.append(out_matches)

    dataset = get_whole_val_set(opt.input_dir, opt.dataset.lower())

    knn = NearestNeighbors(n_jobs=1)
    knn.fit(dataset.dbStruct.locDb)
    _ , gt = knn.radius_neighbors(dataset.dbStruct.locQ,
        radius=dataset.dbStruct.posDistThr)

    predictions = np.zeros((netvlad_preds.shape[0], netvlad_preds.shape[1]))
    best_match_ind = []

    for i in range(predictions.shape[0]):
        predictions[i,:] = netvlad_preds[i, np.flip(np.argsort(match_scores[i,:]))]
        best_match_ind.append(np.flip(np.argsort(match_scores[i,:]))[0])

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

    np.save(os.path.join(output_dir, 'lfm_preds.npy'), predictions)
    np.savetxt(os.path.join(output_dir, 'lfm_recalls.txt'), recall_at_n)

    print('====> Calculating Storage Benefits')
    datasize = 0
    org_datasize = 0
    for i in range(SP_all_db.shape[0]):
        db_topk = topk[i]
        db_cluster_mask = db_cluster_masks[i]
        SP_db = SP_all_db[i]

        db_keypoints = np.flip(SP_db['keypoints'], axis=1)
        db_map = db_cluster_mask[db_keypoints.astype('int')[:, 0].T, db_keypoints.astype('int')[:, 1].T]
        db_filter = np.in1d(db_map, db_topk)
        db_filter_ind = np.where(db_filter == 1)[0]

        datasize += SP_db['descriptors'][:, db_filter_ind].nbytes
        org_datasize += SP_db['descriptors'].nbytes

    storage_percentage = datasize/org_datasize
    print("====> Storage Percentage:{}".format(storage_percentage))

    # Visualize Best Matches along with Utility
    if opt.viz:
        print('====> Visualizing Best Matches along with Utility')
        db_image_paths = dataset.dbStruct.dbImage
        db_image_paths = [x.replace(' ','') for x in db_image_paths]

        query_image_paths = dataset.dbStruct.qImage
        query_image_paths = [x.replace(' ','') for x in query_image_paths]

        matches_viz = []

        for i in range(len(query_image_paths)):
            imageA = imageio.imread(os.path.join(opt.input_dir, query_image_paths[i]))

            # Utility masked Reference Image
            img = skimage.io.imread(os.path.join(opt.input_dir, db_image_paths[predictions[i, 0].astype('int')]))

            db_topk = topk[predictions[i, 0].astype('int')]
            db_cluster_mask = db_cluster_masks[predictions[i, 0].astype('int')]
            utility_binary_mask = np.in1d(db_cluster_mask, db_topk).reshape((480,640))

            plt.axis('off')
            plt.imshow(color.label2rgb(utility_binary_mask, img, colors=[(0,1,1)], bg_label=0, alpha=0.5, kind='overlay'))
            plt.tight_layout(pad=0)

            imageB = fig2img(plt.gcf())
            imageB = imageB.resize((640,480))
            imageB = np.array(imageB)
            plt.close()

            match_result = LFM_matches[(predictions.shape[1]*i + best_match_ind[i]).astype('int')]
            kpsA = match_result['keypoints0']
            kpsB = match_result['keypoints1']
            matches = match_result['matches']

            matched_img = drawMatches(imageA, imageB, kpsA, kpsB, matches)
            matches_viz.append(matched_img)

        imageio.mimsave(os.path.join(output_dir, 'matches.gif'), matches_viz, fps=1)