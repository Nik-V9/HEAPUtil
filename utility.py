from __future__ import print_function
from pathlib import Path
import argparse
import os
import io

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import imageio
import skimage.io
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage import color

from NetVLAD.dataset import get_whole_val_set

parser = argparse.ArgumentParser(description='ES-PS-Utility')
parser.add_argument('--root_dir', type=str, default='', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='berlin', 
        help='Dataset to use', choices=['oxford', 'nordland', 'berlin'])
parser.add_argument('--num_clusters', type=int, default=16, help='Number of NetVlad clusters. Default=16')
parser.add_argument('--netvlad_extracts_path', type=str, default='', help='Path to NetVLAD Extractions')
parser.add_argument('--save_viz', action='store_true', help='Save Utility Visualization Gifs')
parser.add_argument('--save_path', type=str, default='',
        help='Path to save Place-Specific (PS) Utilities & Low Environment-Specific (ES) Utility Clusters')

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    return img

def color_map_color(value, cmap_name='jet', vmin=0, vmax=1):
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb

    return rgb

if __name__ == "__main__":
    opt = parser.parse_args()

    print(opt)

    save_path = Path(opt.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    print("==> Predicting Environment- and Place-Specific Utility")

    dbFeat = np.load(os.path.join(opt.netvlad_extracts_path, 'dbFeat.npy'))
    db_cluster_masks = np.load(os.path.join(opt.netvlad_extracts_path, 'db_cluster_masks.npy'))

    dataset = get_whole_val_set(opt.root_dir, opt.dataset.lower())

    knn = NearestNeighbors(n_jobs=1)
    knn.fit(dataset.dbStruct.locDb)

    _ , positives = knn.radius_neighbors(dataset.dbStruct.locDb,
            radius=dataset.dbStruct.posDistThr, sort_results=True)

    _ , non_negatives = knn.radius_neighbors(dataset.dbStruct.locDb, 
            radius=(2*dataset.dbStruct.posDistThr), sort_results=True)

    ps_utility = np.zeros((dataset.dbStruct.numDb, opt.num_clusters))

    es_utility = np.zeros((opt.num_clusters, 1))

    for j in range(dataset.dbStruct.numDb):
      unique_clusters = np.unique(db_cluster_masks[j]).astype(int)
      indexes = np.arange(dataset.dbStruct.numDb)
      db_neg = np.delete(indexes, non_negatives[j]) # np.delete(x, exclude non-negatives)
      neg = dbFeat[db_neg,:]

      diff = np.zeros((1,opt.num_clusters))
      for i in unique_clusters:
        day_query = dbFeat[j,512*i:512*(i+1)]
        n = neg[:,512*i:512*(i+1)]
        dist = np.linalg.norm(n-day_query, axis=1)

        es_utility[i] += np.sum(dist)

        diff[:,i] = np.average(dist)

      diff = diff+1
      absent_clusters = np.delete(np.arange(opt.num_clusters), unique_clusters)
      diff[:,absent_clusters] = 0
      ps_utility[j] = diff

    es_utility = es_utility/(dataset.dbStruct.numDb*(dataset.dbStruct.numDb-1))

    es_utility_order = np.argsort(es_utility)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(es_utility.reshape(-1, 1))

    if es_utility_order[0] in np.where(kmeans.labels_ == 0)[0]:
        low_es_utility_clusters = np.where(kmeans.labels_ == 0)[0]
    else:
        low_es_utility_clusters = np.where(kmeans.labels_ == 1)[0]

    np.save(os.path.join(opt.save_path, 'ps_utility.npy'), ps_utility)
    np.save(os.path.join(opt.save_path, 'low_es_utility_clusters.npy'), low_es_utility_clusters)

    if opt.save_viz:
        print("==> Visualizing Estimated Utilities")
        image_paths = dataset.dbStruct.dbImage
        image_paths = [x.replace(' ','') for x in image_paths]

        ps_utility_rankings = np.flip(np.argsort(ps_utility, axis=1), axis=1)

        high_es_utility_clusters = np.delete(np.arange(opt.num_clusters), low_es_utility_clusters)

        ps_utility_viz = []
        es_utility_viz = []

        for i in np.arange(len(image_paths))[::4]:
            img = skimage.io.imread(os.path.join(opt.root_dir, image_paths[i]))

            colors = np.zeros((opt.num_clusters,3))
            ps_utility_ranking = ps_utility_rankings[i]
            for j in range(opt.num_clusters):
              colors[ps_utility_ranking[j].astype('int'),:] = color_map_color(j/(opt.num_clusters-1))
            new_colors = colors[np.unique(db_cluster_masks[i]).astype(int)]

            plt.axis('off')
            plt.imshow(color.label2rgb(db_cluster_masks[i], img, colors=new_colors, bg_label=-1, alpha=0.5)) # PS
            plt.tight_layout(pad=0)
            ps_utility_viz.append(fig2img(plt.gcf()))
            plt.close()

            es_binary_mask = np.in1d(db_cluster_masks[i], high_es_utility_clusters).reshape((480,640))

            plt.axis('off')
            plt.imshow(color.label2rgb(es_binary_mask, img, bg_label=1, alpha=0.5)) # ES
            plt.tight_layout(pad=0)
            es_utility_viz.append(fig2img(plt.gcf()))
            plt.close()

        np_ps_utility_viz = [np.array(viz) for viz in ps_utility_viz]
        np_es_utility_viz = [np.array(viz) for viz in es_utility_viz]

        imageio.mimsave(os.path.join(opt.save_path, 'ps_utility.gif'), np_ps_utility_viz, fps=1)
        imageio.mimsave(os.path.join(opt.save_path, 'es_utility.gif'), np_es_utility_viz, fps=1)