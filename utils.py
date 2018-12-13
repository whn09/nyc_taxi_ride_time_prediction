# coding: utf-8
import time
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import radians, cos, sin, asin, sqrt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def rmsle(y_pred, y_test) :
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


def draw_ks(v, n, p, diff, pass_ratio, npass_ratio, ks, ks_x, filename):
    x = v
    plt.figure()
    plt.plot(x, n, '-', color='r', label='N')
    plt.plot(x, p, '-', color='g', label='P')
    plt.plot(x, diff, '-', color='b', label='KS')
    plt.plot(x, pass_ratio, '-', color='c', label=u'Pass')
    plt.plot(x, npass_ratio, '-', color='k', label=u'NPass')
    plt.vlines(ks_x, -1, 1, colors="y", linestyles="dashed")
    plt.xlim(v[0], v[-1])
    plt.ylim(-1, 1)
    plt.legend()
    plt.title('ks=' + str(round(ks, 4)) + ' x=' + str(round(ks_x, 4)))
    plt.savefig(filename)
    # plt.show()


def sort_by_0(s):
    return s[0]


def ks(y_test, y_score, filename):
    if len(y_test) != len(y_score):
        print('ERROR! y_test != y_score')
        return
    v = []
    n = []
    p = []
    diff = []
    pass_ratio = []
    npass_ratio = []
    psum = 0
    nsum = 0
    values = {}
    for i in range(len(y_test)):
        value = int(y_test[i])
        if value == 1:
            psum += 1
        elif value == 0:
            nsum += 1
        else:
            print('ERROR! y_test != 0 or 1')
            return
        if y_score[i] not in values:
            values[y_score[i]] = [0, 0]
        values[y_score[i]][value] += 1

    values_list = []
    for ki, vi in values.items():
        values_list.append([ki, vi[0], vi[1]])
    values_list = sorted(values_list, key=sort_by_0)

    n_pre_sum = 0.0
    p_pre_sum = 0.0
    for value in values_list:
        v.append(value[0])
        try:
            n_rate = (nsum - n_pre_sum) / nsum
        except:
            n_rate = 0
        try:
            p_rate = (psum - p_pre_sum) / psum
        except:
            p_rate = 0
        n.append(n_rate)
        p.append(p_rate)
        diff.append(n_rate - p_rate)
        n_pre_sum += value[1]
        p_pre_sum += value[2]
        pass_ratio.append((nsum - n_pre_sum + psum - p_pre_sum) / (nsum + psum))
        npass_ratio.append((nsum - n_pre_sum) / (nsum + psum))

    ks = max(min(diff), max(diff), key=abs)
    print('ks:', ks)
    ks_x = 0
    for i in range(len(diff)):
        if diff[i] == ks:
            ks_x = v[i]
    # print('v:',len(v))
    # print('n:',len(n))
    # print('p:',len(p))
    # print('diff:',len(diff))
    print('psum:', psum)
    print('nsum:', nsum)
    draw_ks(v, n, p, diff, pass_ratio, npass_ratio, ks, ks_x, filename)
    return ks


def draw_roc(fpr, tpr, roc_auc, filename):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    # plt.show()


def roc(y_test, y_score, filename, draw_image=True):
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)
    if draw_image:
        draw_roc(fpr, tpr, roc_auc, filename)
    return roc_auc


def batch_roc(ids1, ids2, y_test, y_score, filename):
    ids1_stat = {}
    ids2_stat = {}
    for i in range(len(ids1)):
        id1 = ids1[i]
        id2 = ids2[i]
        if id1 not in ids1_stat:
            ids1_stat[id1] = []
        ids1_stat[id1].append([id2, y_test[i], y_score[i]])
        if id2 not in ids2_stat:
            ids2_stat[id2] = []
        ids2_stat[id2].append([id1, y_test[i], y_score[i]])
    ids1_cnt_stat = {}
    ids2_cnt_stat = {}
    for k, v in ids1_stat.items():
        l = len(v)
        if l not in ids1_cnt_stat:
            ids1_cnt_stat[l] = [[], []]
        for vi in v:
            ids1_cnt_stat[l][0].append(vi[1])
            ids1_cnt_stat[l][1].append(vi[2])
    for k, v in ids2_stat.items():
        l = len(v)
        if l not in ids2_cnt_stat:
            ids2_cnt_stat[l] = [[], []]
        for vi in v:
            ids2_cnt_stat[l][0].append(vi[1])
            ids2_cnt_stat[l][1].append(vi[2])
    for k, v in ids1_cnt_stat.items():
        kroc = roc(v[0], v[1], filename[:-4] + '_id1_' + str(k) + filename[-4:], draw_image=False)
        print('ids1_cnt_stat:', k, kroc)
    for k, v in ids2_cnt_stat.items():
        kroc = roc(v[0], v[1], filename[:-4] + '_id2_' + str(k) + filename[-4:], draw_image=False)
        print('ids2_cnt_stat:', k, kroc)


def dump_score(ids, y_test, y_score, filename):
    if len(y_test) != len(ids):
        print('ERROR! y_test != ids')
        return
    if len(y_test) != len(y_score):
        print('ERROR! y_test != y_score')
        return
    fout = open(filename, 'w')
    for i in range(len(y_test)):
        id = ids[i]
        value = y_test[i]
        score = y_score[i]
        fout.write(str(id) + ',' + str(score) + ',' + str(value) + '\n')
    fout.close()


def f1score(y_test, y_score, step=0.1, base=0):
    max_y = int(max(y_test))
    print('max_y:', max_y)
    print('max(y_score):', max(y_score))
    for i in range(0, 10):
        thres = base + i * step
        y_pred = []
        for j in range(len(y_score)):
            for k in range(0, max_y + 1):
                if k == 0:
                    if y_score[j] < k + thres:
                        y_pred.append(k)
                        break
                elif k == max_y:
                    if y_score[j] >= k - 1 + thres:
                        y_pred.append(k)
                        break
                else:
                    if y_score[j] < k + thres and y_score[j] >= k - 1 + thres:
                        y_pred.append(k)
                        break
        print('max(y_pred):', max(y_pred))
        C = confusion_matrix(y_test, y_pred)
        R = classification_report(y_test, y_pred)
        print('thres:', thres, 'confusion_matrix:')
        print(C)
        print('thres:', thres, 'classification_report:')
        print(R)
        TPR_0 = C[0][0] / float(C[0][0] + C[0][1])
        FPR_0 = C[1][0] / float(C[1][0] + C[1][1])
        TPR_1 = C[1][1] / float(C[1][1] + C[1][0])
        FPR_1 = C[0][1] / float(C[0][1] + C[0][0])
        print('thres:', thres, 'TPR_0:', TPR_0, 'FPR_0:', FPR_0)
        print('thres:', thres, 'TPR_1:', TPR_1, 'FPR_1:', FPR_1)


def f1score_one(y_test, y_score, thres=0.5):
    max_y = int(max(y_test))
    y_pred = []
    for j in range(len(y_score)):
        for k in range(0, max_y + 1):
            if k == 0:
                if y_score[j] < k + thres:
                    y_pred.append(k)
                    break
            elif k == max_y:
                if y_score[j] >= k - 1 + thres:
                    y_pred.append(k)
                    break
            else:
                if y_score[j] < k + thres and y_score[j] >= k - 1 + thres:
                    y_pred.append(k)
                    break
    C = confusion_matrix(y_test, y_pred)
    R = classification_report(y_test, y_pred)
    # print('thres:', thres, 'confusion_matrix:')
    print(C)
    # print('thres:', thres, 'classification_report:')
    # print(R)
    TPR_0 = C[0][0] / float(C[0][0] + C[0][1])
    FPR_0 = C[1][0] / float(C[1][0] + C[1][1])
    TPR_1 = C[1][1] / float(C[1][1] + C[1][0])
    FPR_1 = C[0][1] / float(C[0][1] + C[0][0])
    # print('thres:', thres, 'TPR_0:', TPR_0, 'FPR_0:', FPR_0)
    # print('thres:', thres, 'TPR_1:', TPR_1, 'FPR_1:', FPR_1)
    return TPR_0, FPR_0, TPR_1, FPR_1


def opt_tpr_fpr(y_test, y_score, step=0.1, base=0, num=10):
    max_y = int(max(y_test))
    # print('max_y:', max_y)
    # print('max(y_score):', max(y_score))
    TPR_0s = []
    FPR_0s = []
    TPR_1s = []
    FPR_1s = []
    y_score = np.array(y_score)
    start = time.time()
    for i in range(0, num):
        if i % 10 == 0:
            end = time.time()
            print('opt_tpr_fpr:', i, '/', num, end - start)
        thres = base + i * step

        # method 1
        # y_pred = []
        # for j in range(len(y_score)):
        #     for k in range(0, max_y + 1):
        #         if k == 0:
        #             if y_score[j] < k + thres:
        #                 y_pred.append(k)
        #                 break
        #         elif k == max_y:
        #             if y_score[j] >= k - 1 + thres:
        #                 y_pred.append(k)
        #                 break
        #         else:
        #             if y_score[j] < k + thres and y_score[j] >= k - 1 + thres:
        #                 y_pred.append(k)
        #                 break

        # method 2
        def get_y_pred_from_y_score(x, max_y, thres):
            for k in range(0, max_y + 1):
                if k == 0:
                    if x < k + thres:
                        return k
                elif k == max_y:
                    if x >= k - 1 + thres:
                        return k
                else:
                    if k - 1 + thres <= x < k + thres:
                        return k

        get_y_pred = np.frompyfunc(lambda x: get_y_pred_from_y_score(x, max_y, thres), 1, 1)
        y_pred = list(get_y_pred(y_score))

        # print('max(y_pred):', max(y_pred))
        C = confusion_matrix(y_test, y_pred)
        # R = classification_report(y_test, y_pred)
        # print('thres:', thres, 'confusion_matrix:')
        # print(C)
        # print('thres:', thres, 'classification_report:')
        # print(type(R))
        # print(R)
        TPR_0 = C[0][0] / float(C[0][0] + C[0][1])
        FPR_0 = C[1][0] / float(C[1][0] + C[1][1])
        TPR_1 = C[1][1] / float(C[1][1] + C[1][0])
        FPR_1 = C[0][1] / float(C[0][1] + C[0][0])
        # print('thres:', thres, 'TPR_0:', TPR_0, 'FPR_0:', FPR_0)
        # print('thres:', thres, 'TPR_1:', TPR_1, 'FPR_1:', FPR_1)
        TPR_0s.append(TPR_0)
        FPR_0s.append(FPR_0)
        TPR_1s.append(TPR_1)
        FPR_1s.append(FPR_1)

    high_tolerance_fpr = 0.2
    high_tolerance_tpr = 0.45
    medium_tolerance_fpr = 0.1
    medium_tolerance_tpr = 0.35
    low_tolerance_fpr = 0.05
    low_tolerance_tpr = 0.25

    max_high_tolerance_fpr_0 = -1
    max_high_tolerance_tpr_0 = -1
    max_high_tolerance_thres_0 = -1
    max_medium_tolerance_fpr_0 = -1
    max_medium_tolerance_tpr_0 = -1
    max_medium_tolerance_thres_0 = -1
    max_low_tolerance_fpr_0 = -1
    max_low_tolerance_tpr_0 = -1
    max_low_tolerance_thres_0 = -1
    max_high_tolerance_fpr_1 = -1
    max_high_tolerance_tpr_1 = -1
    max_high_tolerance_thres_1 = 2
    max_medium_tolerance_fpr_1 = -1
    max_medium_tolerance_tpr_1 = -1
    max_medium_tolerance_thres_1 = 2
    max_low_tolerance_fpr_1 = -1
    max_low_tolerance_tpr_1 = -1
    max_low_tolerance_thres_1 = 2
    for i in range(0, num):
        thres = base + i * step
        TPR_0 = TPR_0s[i]
        FPR_0 = FPR_0s[i]
        TPR_1 = TPR_1s[i]
        FPR_1 = FPR_1s[i]
        # if FPR_0 <= high_tolerance_fpr and TPR_0 >= high_tolerance_tpr:
        if FPR_0 <= high_tolerance_fpr:
            if FPR_0 > max_high_tolerance_fpr_0:
                max_high_tolerance_fpr_0 = FPR_0
                max_high_tolerance_tpr_0 = TPR_0
                max_high_tolerance_thres_0 = thres
        # if FPR_1 <= high_tolerance_fpr and TPR_1 >= high_tolerance_tpr:
        if FPR_1 <= high_tolerance_fpr:
            if FPR_1 > max_high_tolerance_fpr_1:
                max_high_tolerance_fpr_1 = FPR_1
                max_high_tolerance_tpr_1 = TPR_1
                max_high_tolerance_thres_1 = thres
        # if FPR_0 <= medium_tolerance_fpr and TPR_0 >= medium_tolerance_tpr:
        if FPR_0 <= medium_tolerance_fpr:
            if FPR_0 > max_medium_tolerance_fpr_0:
                max_medium_tolerance_fpr_0 = FPR_0
                max_medium_tolerance_tpr_0 = TPR_0
                max_medium_tolerance_thres_0 = thres
        # if FPR_1 <= medium_tolerance_fpr and TPR_1 >= medium_tolerance_tpr:
        if FPR_1 <= medium_tolerance_fpr:
            if FPR_1 > max_medium_tolerance_fpr_1:
                max_medium_tolerance_fpr_1 = FPR_1
                max_medium_tolerance_tpr_1 = TPR_1
                max_medium_tolerance_thres_1 = thres
        # if FPR_0 <= low_tolerance_fpr and TPR_0 >= low_tolerance_tpr:
        if FPR_0 <= low_tolerance_fpr:
            if FPR_0 > max_low_tolerance_fpr_0:
                max_low_tolerance_fpr_0 = FPR_0
                max_low_tolerance_tpr_0 = TPR_0
                max_low_tolerance_thres_0 = thres
        # if FPR_1 <= low_tolerance_fpr and TPR_1 >= low_tolerance_tpr:
        if FPR_1 <= low_tolerance_fpr:
            if FPR_1 > max_low_tolerance_fpr_1:
                max_low_tolerance_fpr_1 = FPR_1
                max_low_tolerance_tpr_1 = TPR_1
                max_low_tolerance_thres_1 = thres
    return max_high_tolerance_thres_0, max_medium_tolerance_thres_0, max_low_tolerance_thres_0, max_high_tolerance_thres_1, max_medium_tolerance_thres_1, max_low_tolerance_thres_1


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


def bearing_array(lon1, lat1, lon2, lat2):
    lng_delta_rad = np.radians(lon2 - lon1)
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
