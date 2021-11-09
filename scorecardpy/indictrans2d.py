# 二维数据转换测试 
import numpy as np
import pandas as pd
import math
from .woebin import woebin
from .info_value import iv_xy
from .public import iv_threshold



def transfer_2d(x,z,name_x,name_z,y,step=0.1):
    ''' 

    # 尝试进行2D 的数据加权转化处理 
    x: x变量 str
    z: z变量 str
    y: label str
    step: 从-1到1变化的步长,默认0.1 float 0~1  

    return:
    [有效的转化参数名称，[角度,lambda 函数 string 字符串],iv]

    note:
    如果转换后，iv<0.02,直接剔除
    -------------------------------
    Author: 崔超
    '''
    name = ''
    angle_x = -1+step
    df =  pd.DataFrame({name_x:x,name_z:z,'y':y})
    funclist = {}
    funclist[name_x] = [999,f"lambda df: df['{name_x}']"]
    funclist[name_z] = [999,f"lambda df: df['{name_z}']"]

    while angle_x < 1:
        angle_z = 1-abs(int(100*angle_x)/100)
        name = f'({name_x})*{angle_x}+({name_z})*{angle_z}'
        funclist[name]=[angle_x,f"lambda df : df['{name_x}'] * {angle_x} + df['{name_z}'] * {angle_z}"]
        df[name] = eval(funclist[name][1])(df) 
        angle_x += step
    # woe_bin
    bins = woebin(df,y='y')

    # For now, return the best score
    best_trans = sorted([
        [key,[{'trans2d_angle':funclist[key][0]},funclist[key][1]],bins[key].total_iv.values[0]] 
        for key in bins.keys()],key=lambda x: x[2],reverse=True)[0]
    # 剔除有效指标IV 过低或者不需要进行转换的指标
    best_trans = -1 if (best_trans[2]<iv_threshold) or best_trans[1][0]['trans2d_angle']==999 or abs(best_trans[1][0]['trans2d_angle'])< step or abs(best_trans[1][0]['trans2d_angle'])> 1-step else best_trans 
    return best_trans

def transfer_2d_batch(df,x,y,step=0.1,lambda_list=[],x2=[]):
    '''
    # 批量进行指标的2d转化处理
    df: 进行转化的dataframe 
    x: 需要进行转换的列表 list size >2
    x2: 需要进行转换的列表,如果不输入，则为x自己搞 
    y: label str
    step: 见 transfer_2d 
    lambda_list: 初始化的lambda函数列表,
        [有效的转化参数名称，[其他,lambda 函数 string 字符串],iv]

    #
        优化： 对数据首先进行归一化处理
    #
    return:
    list of 见 transfer_2d
    -------------------------------
    Author: 崔超
    '''
    if len(x) <=2:
        raise Exception(' x indics must be list which >2')
    if y in x:
        raise Exception ('BAD! LEAKAGE HAPPENDED')

    # 没有输入就是自己内部搞
    if len(x2)==0:
        x2 = x
    # 首先进行归一化操作
    df_scaled = pd.DataFrame()
    df_scaled[y] =  df[y]
    for i in list(set(x)+set(x2)):
        dmin = df[i].min()
        dmax = df[i].max()
        funcstring = f"lambda df: (df['{i}']-{dmin})/({dmax}-{dmin})"
        lambda_list.append([f"rescale_{i}",[{'trans2d_angle':"angle"},funcstring],-1]) 
        df_scaled[i] = eval(funcstring)[df] 

    transfered = []
    for i in range(len(x)):
        print(f' -- {i}')
        for j in range(len(x2)):
            if x[i] != x2[j] and '--'.join(sorted([x[i],x2[j]])) not in transfered:
                lambda_list.append(transfer_2d(df_scaled[x[i]],df_scaled[x2[j]],x[i],x2[j],df_scaled[y],step))
                transfered.append('--'.join(sorted([x[i],x2[j]])))
    return lambda_list 

