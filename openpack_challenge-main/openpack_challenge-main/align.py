import argparse
import pickle
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import (
    make_submission_zipfile)
from openpack_toolkit.activity import ActSet, ActClass


alpha1 = 0.8
beta = 0.8
joint_path = '/path/to/your/openpack_dataset/v0.3.1/log/openpack-2d-kpt/CTRGCN4Seg/joint/save_scores/joint.pkl'
bone_path = '/path/to/your/openpack_dataset/v0.3.1/log/openpack-2d-kpt/CTRGCN4Seg/bone/save_scores/bone.pkl'
gyro_boundary_path = '/path/to/your/openpack_dataset/v0.3.1/log/atr-left-wrist/DeepConvLSTM/gyro/save_scores/gyro.pkl'
acc_boundary_path = '/path/to/your/openpack_dataset/v0.3.1/log/atr-left-wrist/DeepConvLSTM/acc/save_scores/acc.pkl'
with open(joint_path, 'rb') as r1:
    r1 = pickle.load(r1)
with open(bone_path, 'rb') as r2:
    r2 = pickle.load(r2)

for k in r1:
    r1[k]['y'] = r1[k]['y'] + alpha1 * r2[k]['y']

with open(gyro_boundary_path, 'rb') as r5:
    r5 = pickle.load(r5)

with open(acc_boundary_path, 'rb') as r6:
    r6 = pickle.load(r6)


for k in r5:
    r5[k]['y'] = r5[k]['y'] + r6[k]['y']


def resample_prediction_1Hz(
    ts_unix: np.ndarray = None,
    arr: np.ndarray = None,
):
    """Change the sampling rate into 1 Hz.

    Args:
        ts_unix (np.ndarray): 1d array of unixtimestamp. Defaults to None.
        arr (np.ndarray): 1d array of class IDs. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: arrays of unixtime and resampled class ids.
    """
    assert arr.ndim == 1
    assert ts_unix.ndim == 1
    assert len(arr) == len(
        ts_unix), f"ts_unix={ts_unix.shape}, arr={arr.shape}"

    tmp = ts_unix - (ts_unix % 1000)
    ts_unix_1hz = np.append(tmp, tmp[-1] + 1000)  # FIXME: write in one line

    delta = (ts_unix_1hz[1:] - ts_unix_1hz[:-1])
    assert delta.min() >= 0, (
        "values in array are expected to be monotonically increasing, "
        f"but the minium step is {delta.min()}."
    )

    arr_out, ts_unix_out = [], []
    cur_time = ts_unix_1hz[0]
    for r in range(len(ts_unix_1hz)):
        if cur_time != ts_unix_1hz[r]:
            arr_out.append(arr[r - 1])
            ts_unix_out.append(cur_time)
            cur_time = ts_unix_1hz[r]

    return (
        np.array(ts_unix_out),
        np.array(arr_out),
    )


def construct_submission_dict_align(
        outputs1,
        outputs2,
        act_set: ActSet,
):
    submission = dict()
    for key in outputs1.keys():
        record = dict()

        d1 = outputs1[key]
        d2 = outputs2[key]

        assert (d1["y"].ndim == 3 and d2["y"].ndim == 3)


        ts_unix1 = d1['unixtime'].ravel()
        tmp1 = ts_unix1 - (ts_unix1 % 1000)
        ts_unix_1hz1 = np.append(tmp1, tmp1[-1] + 1000)

        ts_unix2 = d2['unixtime'].ravel()
        tmp2 = ts_unix2 - (ts_unix2 % 1000)
        ts_unix_1hz2 = np.append(tmp2, tmp2[-1] + 1000)


        N,C,L = d1['y'].shape
        score1 = d1['y'].transpose(0,2,1).reshape(-1,C)
        score2 = d2['y'].transpose(0,2,1).reshape(-1,C)


        arr_out1, ts_unix_out1 = [], []
        cur_time1 = ts_unix_1hz1[0]
        for r in range(len(ts_unix_1hz1)):
            if cur_time1 != ts_unix_1hz1[r]:
                arr_out1.append(score1[r - 1])
                ts_unix_out1.append(cur_time1)
                cur_time1 = ts_unix_1hz1[r]

        if key == "U0203-S0100":
            ts_unix_out1.insert(1173, int(1646788845000))
            arr_out1.insert(1173, arr_out1[1172])


        arr_out2, ts_unix_out2 = [], []
        cur_time2 = ts_unix_1hz2[0]
        for r in range(len(ts_unix_1hz2)):
            if cur_time2 != ts_unix_1hz2[r]:
                arr_out2.append(score2[r - 1])
                ts_unix_out2.append(cur_time2)
                cur_time2 = ts_unix_1hz2[r]



        index = 0
        arr_out1 = np.array(arr_out1)
        arr_out2 = np.array(arr_out2)

        if ts_unix_out2[0] <= ts_unix_out1[0]:
            for i, t in enumerate(ts_unix_out2):
                if t == ts_unix_out1[0]:
                    index = i
                    break
            if len(arr_out2[index:])<=len(arr_out1):
                arr_out1[:len(arr_out2[index:])] = arr_out1[:len(arr_out2[index:])] + arr_out2[index:]*beta
                check = (ts_unix_out2[:len(arr_out2[index:])] == ts_unix_out2[index:])

                assert check==True

            else:
                arr_out1 = arr_out1 + arr_out2[index:len(arr_out1)+index]*beta
                check = (ts_unix_out1 == ts_unix_out2[index:len(arr_out1)+index])
                assert check==True

        else:
            for i, t in enumerate(ts_unix_out1):
                if t == ts_unix_out2[0]:
                    index = i
                    break

            if len(arr_out1[index:])<=len(arr_out2):
                arr_out1[index:] = arr_out1[index:]  + arr_out2[:len(arr_out1[index:])]*beta
                check = (ts_unix_out1[index:] ==ts_unix_out2[:len(arr_out1[index:])])
                assert check == True

            else:
                arr_out1[index:len(arr_out2)+index] = arr_out1[index:len(arr_out2)+index] + arr_out2*beta
                check = (ts_unix_out1[index:len(arr_out2)+index] == ts_unix_out2)
                assert check == True





        arr_out1 = np.argmax(arr_out1, axis=1)
        unixtime_pred_sess = np.array(ts_unix_out1)

        prediction_sess = act_set.convert_index_to_id(
            arr_out1)

        record["unixtime"] = unixtime_pred_sess.copy()
        record["prediction"] = prediction_sess.copy()
        submission[key] = record

    return submission




if __name__ == '__main__':
    submission_dict = construct_submission_dict_align(r1, r5, OPENPACK_OPERATIONS)

    metadata = {'dataset.split.name': 'openpack-challenge-2022'}
    logdir = Path('./result')

    make_submission_zipfile(submission_dict, logdir, metadata=metadata)



