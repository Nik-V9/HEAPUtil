from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd

from NetVLAD.dataset import get_whole_val_set

parser = argparse.ArgumentParser(description='Generate Data Path Lists for SuperPoint and SuperGlue')
parser.add_argument('--root_dir', type=str, default='', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='berlin', 
        help='Dataset to use', choices=['oxford', 'nordland', 'berlin'])
parser.add_argument('--netvlad_predictions', type=str, default='', help='Path to NetVLAD Predictions')
parser.add_argument('--save_path', type=str, default='', help='Path to save the data path lists')

if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)

    save_path = Path(opt.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    print("==> Generating Path Lists")

    dataset = get_whole_val_set(opt.root_dir, opt.dataset.lower())

    db_lst = dataset.dbStruct.dbImage
    db_lst = [x.replace(' ','') for x in db_lst]
    q_lst = dataset.dbStruct.qImage
    q_lst = [x.replace(' ','') for x in q_lst]

    outfile = os.path.join(opt.save_path, 'db_list.txt')

    with open(outfile, "w") as outfile:
        outfile.write("\n".join(str(item) for item in db_lst))

    outfile = os.path.join(opt.save_path, 'q_list.txt')

    with open(outfile, "w") as outfile:
        outfile.write("\n".join(str(item) for item in q_lst))

    netvlad_candidates = np.load(opt.netvlad_predictions)

    candidate_list = []
    for i in range(netvlad_candidates.shape[0]):
      for j in range(netvlad_candidates.shape[1]):
        candidate_list.append([q_lst[i], db_lst[netvlad_candidates[i,j]]])

    df = pd.DataFrame.from_records(candidate_list)
    df.to_csv(os.path.join(opt.save_path, opt.dataset+'_netvlad_candidate_list.txt'), header=None, index=None, sep=' ', mode='a')