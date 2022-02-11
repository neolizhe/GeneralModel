#coding:utf-8
import pandas as pds
import sys
work_path = '/home/strategy_04/neolizhe/data/'

if __name__=="__main__":
    file_name = sys.argv[1]
    dfs = pds.read_csv(work_path + file_name, iterator=True, chunksize=10**7)
    res_map = {}
    index = 0
    for df in dfs:
        #cnt, min, max, mean, std
        # cnt = cnt1 + cnt2;
        # min = min1 < min2  ? min1 : min2;
        # max = max1 > max2 ? max1 : max2;
        # mean = (mean1*cnt1 + mean2*cnt2)/(cnt1+cnt2)
        # std = s1^2*cnt1+s2^2*cnt2 + (mean1-mean)^2*cnt1 + (mean2 -mean)^2*cnt2
        # std = std / cnt
        # std = sqrt(std)
        df = df.fillna(0)
        if index == 0:
            cols = df.columns
        for col in cols:
            if index == 0:
                cnt = len(df[col])
                minv = df[col].min()
                maxv = df[col].max()
                meanv = df[col].mean()
                stdv = df[col].std()
                res_map[col]  = [cnt, minv, maxv, meanv, stdv]
            else:
                cnt = res_map[col][0] + len(df[col])
                minv = res_map[col][1] if res_map[col][1] < df[col].min() else df[col].min()
                maxv = res_map[col][2] if res_map[col][2] > df[col].max() else df[col].max()
                tmp_meanv = (df[col].sum() + res_map[col][3] * (cnt - len(df[col])))/cnt
                stdv = (res_map[col][4]**2 + (res_map[col][4] - tmp_meanv)**2)*res_map[col][0] + (df[col].std()**2 + (df[col].mean() - tmp_meanv)**2)*len(df[col])
                stdv = (stdv/cnt)**0.5
                meanv = tmp_meanv
                res_map[col]  = [cnt, minv, maxv, meanv, stdv]
        index += 1
        print("iters %s" % index)
    f=open('normal_params.txt','w')
    f.write("columns"+"\t"+"cnt\t"+"min\t"+"max\t"+"mean\t"+"std\t \n")
    for k,v in res_map.items():
        f.write(k + ',')
        v = [str(x) for x in v]
        f.write(','.join(v))
        f.write('\n')
    print("file write done!")
    f.close()
