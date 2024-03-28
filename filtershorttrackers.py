import pandas as pd
import os
import copy

if __name__ == '__main__':
    resultpath = './transformed2D'
    for i in range(26, 27):
        file = os.path.join(resultpath, 'new2Dposresult' + str(i) + '.txt')
        df = pd.read_table(file, sep=',')
        last = int(df.values[-1][0])
        res1 = df[df.cam == 1]
        res2 = df[df.cam == 2]
        res3 = df[df.cam == 3]
        ids1 = res1['ID'].unique()
        ids2 = res2['ID'].unique()
        ids3 = res3['ID'].unique()
        res1_1 = copy.deepcopy(res1)
        res2_1 = copy.deepcopy(res2)
        res3_1 = copy.deepcopy(res3)
        for id in ids1:
            trackerlength = len(res1[res1.ID == id])
            if trackerlength <= 3:
                res1_1 = res1_1[res1_1.ID != id]
        for id in ids2:
            trackerlength = len(res2[res2.ID == id])
            if trackerlength <= 3:
                res2_1 = res2_1[res2_1.ID != id]
        for id in ids3:
            trackerlength = len(res3[res3.ID == id])
            if trackerlength <= 3:
                res3_1 = res3_1[res3_1.ID != id]
        result_df = pd.concat([res1_1, res2_1, res3_1], axis=0)
        result_df.to_csv(os.path.join(resultpath, 'newfiltershort2Dposresult' + str(i) + '.txt'), index=False, sep=',')
