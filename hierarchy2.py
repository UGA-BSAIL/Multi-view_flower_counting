from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist, squareform
import pandas as pd
import os
import cv2
import copy


def constrained_hierarchical_clustering(X, cl=[], thres=30, linkage_method='complete'):
    D = cdist(X, X)
    for (i, j) in cl:
        D[i, j] = 10 ** 8
        D[j, i] = 10 ** 8
    Z = linkage(squareform(D), method=linkage_method)
    clusters = fcluster(Z, t=thres, criterion='distance')
    return clusters


if __name__ == "__main__":
    indexx = 26
    plot = [[6900, 7152]]

    for plo in plot:
        path = './transformed2D/new2Dposresult' + str(indexx) + '.txt'
        dat = '20220907103911428290'
        imgpath = '../img/color_img2_' + dat
        savepath = './clustered/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        save_img = True
        save_text = True
        # colours = np.random.rand(32, 3) * 255
        colours = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 127, 0), (255, 0, 0), (255, 0, 139),
                   (127, 0, 255), (127, 127, 0), (0, 127, 127), (75, 75, 255), (255, 75, 75), (75, 255, 75),
                   (286, 211, 32), (28, 165, 230), (165, 50, 220)]
        df = pd.read_table(path, sep=',')
        if len(df) != 0:
            last = df['frame'].max()
            match12 = {}
            match13 = {}
            match23 = {}
            unmatch1 = []
            unmatch2 = []
            unmatch3 = []
            remain1 = []
            remain2 = []
            remain3 = []
            for i in range(1, last + 1):
                ml = []
                concat = []
                conid = []
                res1 = df[df.frame == i]
                imgname = "frame" + str(i + plo[0]) + ".jpg"
                img = cv2.imread(os.path.join(imgpath, imgname))
                set1 = res1[res1.cam == 1]
                set2 = res1[res1.cam == 2]
                set3 = res1[res1.cam == 3]
                if len(set1.values) > 1:
                    for m in range(len(set1.values)):
                        for n in range(m + 1, len(set1.values)):
                            ml.append([m, n])
                if len(set2.values) > 1:
                    for m in range(len(set2.values)):
                        for n in range(m + 1, len(set2.values)):
                            ml.append([m + len(set1.values), n + len(set1.values)])
                if len(set3.values) > 1:
                    for m in range(len(set3.values)):
                        for n in range(m + 1, len(set3.values)):
                            ml.append(
                                [m + len(set1.values) + len(set2.values), n + len(set1.values) + len(set2.values)])

                for item in set1.values:
                    x, y = item[5], item[6]
                    id = item[2]
                    concat.append([x, y])
                    conid.append(id)
                for item in set2.values:
                    x, y = item[5], item[6]
                    id = item[2]
                    concat.append([x, y])
                    conid.append(id)
                for item in set3.values:
                    x, y = item[5], item[6]
                    id = item[2]
                    concat.append([x, y])
                    conid.append(id)

                if len(concat) > 1:
                    result = constrained_hierarchical_clustering(concat, cl=ml, thres=60, linkage_method="complete")
                    a = result[set1.shape[0]:set1.shape[0] + set2.shape[0]]
                    for p in range(set1.shape[0]):
                        if result[p] in result[set1.shape[0]:set1.shape[0] + set2.shape[0]]:  # cam1-2
                            if conid[p] not in match12:
                                match12[conid[p]] = [conid[list(
                                    result[set1.shape[0]:set1.shape[0] + set2.shape[0]]).index(result[p]) + set1.shape[
                                                               0]]]
                            if conid[p] in match12 and conid[
                                list(result[set1.shape[0]:set1.shape[0] + set2.shape[0]]).index(result[p]) + set1.shape[
                                    0]] not in match12[conid[p]]:
                                match12[conid[p]].append(conid[list(
                                    result[set1.shape[0]:set1.shape[0] + set2.shape[0]]).index(result[p]) + set1.shape[
                                                                   0]])
                        if result[p] in result[set1.shape[0] + set2.shape[0]:]:  # cam1-3
                            if conid[p] not in match13:
                                match13[conid[p]] = [
                                    conid[
                                        list(result[set1.shape[0] + set2.shape[0]:]).index(result[p]) + set1.shape[0] +
                                        set2.shape[0]]]
                            if conid[p] in match13 and conid[
                                list(result[set1.shape[0] + set2.shape[0]:]).index(result[p]) + set1.shape[0] +
                                set2.shape[0]] not in match13[conid[p]]:
                                match13[conid[p]].append(conid[list(result[set1.shape[0] + set2.shape[0]:]).index(
                                    result[p]) + set1.shape[0] + set2.shape[0]])
                        if result[p] not in result[set1.shape[0]:set1.shape[0] + set2.shape[0]] and result[
                            p] not in result[set1.shape[0] + set2.shape[0]:] and conid[p] not in match12 and conid[
                            p] not in match13:  # cam1
                            if conid[p] not in unmatch1:
                                unmatch1.append(conid[p])
                    for q in range(set2.shape[0]):
                        if result[set1.shape[0]:][q] in result[set1.shape[0] + set2.shape[0]:]:  # cam2-3
                            if conid[set1.shape[0]:][q] not in match23:
                                match23[conid[set1.shape[0]:][q]] = [conid[list(
                                    result[set1.shape[0] + set2.shape[0]:]).index(result[set1.shape[0]:][q]) +
                                                                           set1.shape[0] + set2.shape[0]]]
                            if conid[set1.shape[0]:][q] in match23 and conid[
                                list(result[set1.shape[0] + set2.shape[0]:]).index(result[set1.shape[0]:][q]) +
                                set1.shape[0] + set2.shape[0]] not in match23[conid[set1.shape[0]:][q]]:
                                match23[conid[set1.shape[0]:][q]].append(conid[list(
                                    result[set1.shape[0] + set2.shape[0]:]).index(result[set1.shape[0]:][q]) +
                                                                               set1.shape[0] + set2.shape[0]])
                        if result[set1.shape[0]:][q] not in result[set1.shape[0] + set2.shape[0]:] and \
                                result[set1.shape[0]:][q] not in result[:set1.shape[0]]:
                            if conid[set1.shape[0]:][q] not in unmatch2:
                                unmatch2.append(conid[set1.shape[0]:][q])
                    for s in range(set3.shape[0]):
                        if result[set1.shape[0] + set2.shape[0]:][s] not in result[:set1.shape[0] + set2.shape[0]]:
                            if conid[set1.shape[0] + set2.shape[0]:][s] not in unmatch3:
                                unmatch3.append(conid[set1.shape[0] + set2.shape[0]:][s])

                    ind = 0
                    for (x, y) in concat:
                        labels = result[ind]
                        color = colours[labels]
                        cv2.circle(img, (int(x), int(y)), 5, color, -1)
                        cv2.putText(img, str(int(conid[ind])), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                    1)
                        ind += 1
                elif len(concat) == 1:
                    if len(set1.values) == 1:
                        if conid[0] not in remain1:
                            remain1.append(conid[0])
                    if len(set2.values) == 1:
                        if conid[0] not in remain2:
                            remain2.append(conid[0])
                    if len(set3.values) == 1:
                        if conid[0] not in remain3:
                            remain3.append(conid[0])
                    color = colours[0]
                    cv2.circle(img, (int(concat[0][0]), int(concat[0][1])), 5, color, -1)
                    cv2.putText(img, str(int(conid[0])), (int(concat[0][0]), int(concat[0][1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

                cv2.imshow('img', img)
                if save_img == True:
                    cv2.imwrite(savepath + str(i + plo[0]) + '.jpg', img)
                cv2.waitKey(10)

            newunmatch1 = copy.deepcopy(unmatch1)
            for each in unmatch1:
                if each in match12.keys() or each in match13.keys():
                    newunmatch1.remove(each)
            print(newunmatch1)
            newunmatch2 = copy.deepcopy(unmatch2)
            for each in unmatch2:
                if each in match23.keys():
                    newunmatch2.remove(each)
            newunmatch2_1 = copy.deepcopy(newunmatch2)
            for each in newunmatch2:
                for key in match12.keys():
                    if each in match12[key] and each in newunmatch2_1:
                        newunmatch2_1.remove(each)
            print(newunmatch2_1)
            newunmatch3 = copy.deepcopy(unmatch3)
            for each in unmatch3:
                for key in match13.keys():
                    if each in match13[key] and each in newunmatch3:
                        newunmatch3.remove(each)
            newunmatch3_1 = copy.deepcopy(newunmatch3)
            for each in newunmatch3:
                for key in match23.keys():
                    if each in match23[key] and each in newunmatch3_1:
                        newunmatch3_1.remove(each)
            print(newunmatch3_1)
            number = 0
            for key in match13.keys():
                for key12 in match12.keys():
                    if key == key12:
                        number += 1
            for key in match23.keys():
                for key12 in match12.keys():
                    if key in match12[key12]:
                        number += 1
            newremain1 = copy.deepcopy(remain1)
            newremain2 = copy.deepcopy(remain2)
            newremain3 = copy.deepcopy(remain3)
            for idx in remain1:
                if idx in match13.keys() or idx in match12.keys() or idx in unmatch1:
                    newremain1.remove(idx)
            for idx in remain2:
                if idx in match23.keys() or idx in unmatch2:
                    newremain2.remove(idx)
                for key in match12.keys():
                    if idx in match12[key] and idx in newremain2:
                        newremain2.remove(idx)
            for idx in remain3:
                if idx in unmatch3:
                    newremain3.remove(idx)
                for key in match13.keys():
                    if idx in match13[key] and idx in newremain3:
                        newremain3.remove(idx)
                for key in match23.keys():
                    if idx in match23[key] and idx in newremain3:
                        newremain3.remove(idx)

            print('***********')
            counts = len(match12.keys()) + len(match13.keys()) + len(match23.keys()) + len(newunmatch1) + len(
                newunmatch2_1) + len(newunmatch3_1) - number + len(newremain1) + len(newremain2) + len(newremain3)
            print("final count is %d." % counts)
            if save_text == True:
                with open(savepath + 'final counts 0922.txt', 'a+') as f:
                    f.write('video %d has %d. flowers \n' % (indexx, counts))
            indexx += 1
        else:
            print('***********')
            print("final count is %d." % 0)
            if save_text == True:
                with open(savepath + 'final counts 0922.txt', 'a+') as f:
                    f.write('video %d has %d. flowers \n' % (indexx, 0))
            indexx += 1
