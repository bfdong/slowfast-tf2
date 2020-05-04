import time
import numpy as np
from tqdm import tqdm,trange

def topk(predicts, labels, ids):
    scores = {}
    top1_list = []
    top5_list = []
    clips_top1_list = []
    clips_top5_list = []
    start_time = time.time()
    print('Results process..............')
    for index in tqdm(range(len(predicts))):
        id = ids[index]
    score = predicts[index]
    if str(id) not in scores.keys():
        scores['%d'%id] = []
        scores['%d'%id].append(score)
    else:
        scores['%d'%id].append(score)
    avg_pre_index = np.argsort(score).tolist()
    top1 = (labels[id] in avg_pre_index[-1:])
    top5 = (labels[id] in avg_pre_index[-5:])
    clips_top1_list.append(top1)
    clips_top5_list.append(top5)
    print('Clips-----TOP_1_ACC in test: %f' % np.mean(clips_top1_list))
    print('Clips-----TOP_5_ACC in test: %f' % np.mean(clips_top5_list))
    print('..............')
    for _id in range(len(labels)-1):
        avg_pre_index = np.argsort(np.mean(scores['%d'%_id], axis=0)).tolist()
    top1 = (labels[_id] in avg_pre_index[-1:])
    top5 = (labels[_id] in avg_pre_index[-5:])
    top1_list.append(top1)
    top5_list.append(top5)
    print('TOP_1_ACC in test: %f' % np.mean(top1_list))
    print('TOP_5_ACC in test: %f' % np.mean(top5_list))
    duration = time.time() - start_time
    print('Time use: %.3f' % duration)