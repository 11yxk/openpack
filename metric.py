
import numpy as np


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1])
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    print(111111)
    print(recognized.shape)

    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    print(len(p_label))

    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
    print(len(y_label))
    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    # print(p_label)
    print(p_start)
    print(p_end)
    # print(y_label)
    print(y_start)
    print(y_end)
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        print(np.minimum(p_end[j], y_end))
        print(np.maximum(p_start[j], y_start))
        print(intersection)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        print(np.maximum(p_end[j], y_end))
        print(np.minimum(p_start[j], y_start))
        print(union)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        print((1.0 * intersection / union))
        print(([p_label[j] == y_label[x] for x in range(len(y_label))]))
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
        print(IoU)


        print(idx)
        print(np.array(IoU).shape)
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


if __name__ == '__main__':
    import torch
    import os
    from tqdm import tqdm

    path ='/mnt/data3/chai/openpack_dataset/journal_v0.3.1/log/openpack-2d-kpt/CTRGCN4Seg/CTRGCN_base/pred'

    pred = []
    label = []
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    overlap = [.1, .25, .5]
    correct = 0
    total = 0
    edit = 0
    flap = 0
    for item in tqdm(os.listdir(path)):
        if flap ==1:
            break
        flap = 1
        gt_content = np.load(os.path.join(path, item, 't.npy')).reshape(-1)
        recog_content = np.load(os.path.join(path, item, 'y.npy'))
        recog_content = np.argmax(recog_content,axis=1).reshape(-1)
        print(recog_content.shape,gt_content.shape)

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(os.listdir(path))
    print('Accyracy: %.4f' % (acc))
    print('edit: %.4f' % (edit))
    f1s = np.array([0, 0, 0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1

