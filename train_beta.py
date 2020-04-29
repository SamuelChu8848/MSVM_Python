import pandas as pd
import numpy as np
import os
import sys  #读取文件相对位置
import xlrd
from sklearn.preprocessing import MinMaxScaler      #归一化使用
import copy
from sklearn.model_selection import train_test_split        #划分数据集使用
from sklearn.metrics import r2_score                #计算r2_score用
from sklearn.metrics import mean_squared_error      #计算RMSE用
from math import sqrt

import random
import math
import xlwings as xw
import time
import eventlet     #导入eventlet这个模块用于超时

outtime_count = 0
'''
extraction_collocation
(函数测试通过)
从往期数据中24小时每个时刻使用过的搭配

Inputs: 
    data: 往期数据（n条*2，时间，搭配）data为数组

Output:
    all_hour_collocation: 提取出的搭配列表 24*24
    collocation_num: 每一时刻搭配种类数1*24

'''
def extraction_collocation(data):

    all_hour_collocation = np.zeros((24,50))    #初始化24小时泵组搭配种类表
    (row,col) = data.shape      #获取数组行数、列数
    collocation_num = []        #初始化每个小时泵组搭配采用情况的种类数，例1时刻有10种类搭配

    for hour in range(1,25):        #对24个时刻循环，求其搭配种类
        count = 1       
        for num_data in range(1,row+1):     #对数据集一一对比
            one_hour_collocation = all_hour_collocation[hour-1]
            time = data[num_data-1,0]
            collocation = data[num_data-1,1]
            if (hour == time) and (collocation not in one_hour_collocation):
                all_hour_collocation[hour-1,count-1] = collocation
                count += 1
        collocation_num.append(count-1)
    return all_hour_collocation,collocation_num



'''
getsheet
（测试通过）(未使用)
从excel获得某一工作簿数据

Inputs: 
    file_name: 与.py文件同一目录下的文件名，字符型,带后缀(*.xls)
    sheet_name: 指定工作簿，字符型

Output:
    table_get：指定sheet

'''
def getsheet(file_name,sheet_name):
    path = os.path.abspath(os.path.dirname(sys.argv[0]))    #py文件路径
    excelPath = os.path.join(path,file_name)
    data = xlrd.open_workbook(excelPath)
    table_get = data.sheet_by_name(sheet_name)
    return table_get


'''
printlist

按行列输出list类型数据

Inputs: 
    arr: list列表

Output:
    打印
'''
def printlist(arr):

    for i in range(len(arr)):      # 控制行，0~2
        for j in range(len(arr[i])):    # 控制列
            print(arr[i][j], end='\t')
        print()


'''
dataselect

从数据集data中筛选出某一时刻某一搭配的数据集
如果数据集少于10条，则提取本搭配下，所有时刻的数据作为数据集
如果数据集少于5条，则放弃使用泵搭配
在对某一时刻，各类搭配的循环求解时中使用

Inputs: 
    data: 源数据集
    timeset: 设定时刻
    collocationset：设定搭配

Output:
    X：符合条件的输入数据
    Y：符合条件的输出数据
    interrupt: 0 正常执行；1 结束主函数中本次循环

'''
def dataselect(data,timeset,collocationset):
    interrupt = 0
    data =  data.tolist()
    X = []
    Y = []
    if collocationset == 0:      #每个时刻，遇0则表示所有搭配都已预测完成,结束本次循环
        interrupt = 1
   
    for data_eachrow in data:
        if (data_eachrow[0] == timeset) and (data_eachrow[1] == collocationset):
            X.append(data_eachrow[2:7])
            Y.append(data_eachrow[7:])
    
    if len(X) < 10:    #数据小于10条，从24小时提取搭配
        X = []
        Y = []
        for data_eachrow in data:
            if data_eachrow[1] == collocationset:
                X.append(data_eachrow[2:7])
                Y.append(data_eachrow[7:])
    
    if len(X) < 5:      #数据仍小于5条，放弃本组搭配
        interrupt = 1

    return X,Y,interrupt


'''
msvr

inputs:
    x : training patterns (num_samples * n_d),
    y : training targets (num_samples * n_k),
    ker : kernel type ('lin', 'poly', 'rbf'),
    C : cost parameter,
    par : kernel parameter (see function 'kernelmatrix'),
    tol : tolerance.
Outputs:
    Beta
    outtime_flag
'''
def msvr(x, y, ker, C, epsi, par, tol):
    global outtime_count
    global outtime_flag

    n_m = np.shape(x)[0]    #样本数量
    n_d = np.shape(x)[1]    #输入变量维度
    n_k = np.shape(y)[1]    #输出变量维度
    
    H = kernelmatrix(ker, x, x, par)    #创建核矩阵
    Beta = np.zeros((n_m, n_k))    #创建回归参数
    
    E = y - np.dot(H, Beta)    #E为每个输出的预测误差 (n_m * n_k)
  
    u = np.sqrt(np.sum(E**2,1,keepdims=True))     #RSE
    
    RMSE = []
    RMSE_0 = np.sqrt(np.mean(u**2))
    RMSE.append(RMSE_0)      #RMSE
    
   
    i1 = np.where(u>epsi)[0]     #预测误差大于epsilon的点

    a = 2 * C * (u - epsi) / u    #设置alphas的初始值a(n_m * 1)
    
    L = np.zeros(u.shape)   #L (n_m * 1)
    
    L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2   #仅对 u > epsi使用sq 松弛因子计算
    
    Lp = []     #Lp 是要求最小的数值 （sq为参数+松弛因子）
    BetaH = np.dot(np.dot(Beta.T, H), Beta)
    Lp_0 = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
    Lp.append(Lp_0)
    
    eta = 1
    k = 1
    hacer = 1
    val = 1
    
    while(hacer and outtime_count < 20000):     #加入计次函数，次数过多则跳过
        Beta_a = Beta.copy()
        E_a = E.copy()
        u_a = u.copy()
        i1_a = i1.copy()
        
        M1 = H[i1][:,i1] + np.diagflat(1/a[i1]) + 1e-10 * np.eye(len(a[i1]))
        
        #计算batas
        sal1 = np.dot(np.linalg.inv(M1),y[i1]) #求逆or广义逆（M-P逆）无法保证M1一定是可逆的？
        
        eta = 1
        Beta = np.zeros(Beta.shape)
        Beta[i1] = sal1.copy()
        
        E = y - np.dot(H, Beta)        #偏差

        u = np.sqrt(np.sum(E**2,1)).reshape(n_m,1)      #RSE
        i1 = np.where(u>=epsi)[0]
        
        L = np.zeros(u.shape)
        L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
        
        #重新计算损失函数
        BetaH = np.dot(np.dot(Beta.T, H), Beta)
        Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
        Lp.append(Lp_k)
        
        #循环保存alphas并修改betas
        while(Lp[k] > Lp[k-1]):
            eta = eta/10
            i1 = i1_a.copy()
            
            Beta = np.zeros(Beta.shape)       #新的beta是当前 (sal1)的组合 
          
            Beta[i1] = eta*sal1 + (1-eta)*Beta_a[i1]    #上一次迭代
            
            E = y - np.dot(H, Beta)
            u = np.sqrt(np.sum(E**2,1)).reshape(n_m,1)

            i1 = np.where(u>=epsi)[0]
            
            L = np.zeros(u.shape)
            L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
            BetaH = np.dot(np.dot(Beta.T, H), Beta)
            Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
            Lp[k] = Lp_k
            
            #停止标准1
            if(eta < 1e-16):
                Lp[k] = Lp[k-1]- 1e-15
                Beta = Beta_a.copy()
                
                u = u_a.copy()
                i1 = i1_a.copy()
                
                hacer = 0
        
        #修改alpha并保留betas
        a_a = a.copy()
        a = 2 * C * (u - epsi) / u
        
        RMSE_k = np.sqrt(np.mean(u**2))
        RMSE.append(RMSE_k)
    
        if((Lp[k-1]-Lp[k])/Lp[k-1] < tol):
            hacer = 0

        outtime_count += 1
        if outtime_count%1000 == 0:
            print('============监测点==========',outtime_count)

        k = k + 1
        
        #停止标准，算法不收敛 (val = -1)
        if(len(i1) == 0):
            hacer = 0
            Beta = np.zeros(Beta.shape)
            val = -1
            
    NSV = len(i1)
    
    if outtime_count >= 20000:      #给出超时信号beta = 0
        outtime_flag = 1

    return Beta


'''
kernelmatrix

从训练和测试数据矩阵构建核矩阵

Inputs: 
    ker: {'lin' 'poly' 'rbf'}
    X: Xtest (num_test * n_d)
    X2: Xtrain (num_train * n_d)
    parameter: 
       width of the RBF kernel
       bias in the linear and polinomial kernel 
       degree in the polynomial kernel

Output:
    K: 核矩阵
'''
def kernelmatrix(ker, X, X2, p=0):
    X = np.array(X)
    X2 = np.array(X2)
    X = X.T
    X2 = X2.T

    if(ker == 'lin'):
        tmp1, XX2_norm, tmp2 = np.linalg.svd(np.dot(X.T,X2))
        XX2_norm = np.max(XX2_norm)
        K = np.dot(X.T,X2)/XX2_norm
    
    elif(ker == 'poly'):
        tmp1, XX2_norm, tmp2 = np.linalg.svd(np.dot(X.T,X2))
        XX2_norm = np.max(XX2_norm)
        K = (np.dot(X.T,X2)/XX2_norm*p[0] + p[1]) ** p[2]
    
    elif(ker == 'rbf'): 
        n1sq = np.sum(X**2,0,keepdims=True)
        n1 = X.shape[1]
        
        if(n1 == 1):        #仅一个特征
            N1 = X.shape[0]
            N2 = X2.shape[0]
            D = np.zeros((N1,N2))
            for i in range(0,N1):
                D[i] = (X2 - np.dot(np.ones((N2,1)),X[i].reshape(1,-1))).T * (X2 - np.dot(np.ones((N2,1)),X[i].reshape(1,-1))).T
        else:
            n2sq = np.sum(X2**2,0,keepdims=True)
            n2 = X2.shape[1]
            D = (np.dot(np.ones((n2,1)),n1sq)).T + np.dot(np.ones((n1,1)),n2sq) - 2*np.dot(X.T, X2)
        
        K = np.exp((-D)/(2*p**2))
        
    else:
        print("no such kernel")
        K = 0
        
    return K


'''
fit_fun

粒子群优化算法的适应度函数

Inputs:
    present：初始位置
    train_X: 输入训练集
    train_Y: 输出训练集
    test_X: 输入测试集
    test_Y: 输出测试集

Output:
    fit：适应度
'''
def fit_fun(present,X,Y,Xt,Yt):
    ker  = 'rbf'
    tol  = 1e-20
    C=present[0]
    par=present[1]
    epsi=present[2]
    #print('pop送入适应度计算的值：',present)

    beta = msvr(X,Y,ker,C,epsi,par,tol)

    Ktest = kernelmatrix(ker,Xt,X,par)

    Ypredtest = np.dot(Ktest,beta)
    #print(Ypredtest)
    fit = ((Yt - Ypredtest)**2).sum()
    #print(fit)

    return fit

'''
psoformsvr
【存在可改进地方】
利用粒子群优化算法优化参数

Inputs: 
    train_X: 输入训练集
    train_Y: 输出训练集
    test_X: 输入测试集
    test_Y: 输出测试集

Output:
    global_x:最优解
    fit_gen:迭代寻优
'''
def psoformsvr(X,Y,Xt,Yt):

    global outtime_flag
    c1 = 0.9
    c2 = 0.9
    maxgen = 100
    sizepop = 20
    k = 0.6
    wV = 0.9
    wP = 0.9
    popcmax = 10**3
    popcmin = 10**(-3) 
    popgmax = 10**3
    popgmin = 10**(-3)
    pophmax = 10**3
    pophmin = 10**(-3)

    Vcmax = k * popcmax
    Vcmin = -Vcmax
    Vgmax = k * popgmax
    Vgmin = -Vgmax
    Vhmax = k * popgmax
    Vhmin = -Vgmax

    #随机产生初始粒子和速度                 【】
    pop = []
    V = []
    fitness = []
    for i in range(1,sizepop+1):
        pop.append([ (popcmax-popcmin)*random.random()+popcmin, (popgmax-popgmin)*random.random()+popgmin, (pophmax-pophmin)*random.random()+pophmin])
        V.append([Vcmax*random.random(), Vgmax*random.random(), Vhmax*random.random()])
        fitness.append([fit_fun(pop[i-1],X,Y,Xt,Yt)])
    
    #print('适应度数组：\n',np.array(fitness))

    #找极值和极值点
    global_fitness = min(fitness)       #全局极值点
    bestindex = fitness.index(min(fitness))     #极值点序号
    local_fitness = list(fitness)       #个体极值初始 (单独创建内存空间)

    global_x = pop[bestindex]       #全局极值点     学习心得：引用列表内容创建新的内存空间
    local_x = pop[:]        #个体极值点初始化       学习心得：此处和list创建内存空间作用相同
    avgfitness_gen = [0]*maxgen     #初始化每代的平均适应度

    #迭代寻优
    fit_gen = []
    exit_flag = False
    for generation in range(1,maxgen+1):
        for eachpop in range(1,sizepop+1):
            #速度更新
            weight1 = np.array(V[eachpop-1])
            weight2 = np.array(local_x[eachpop-1]) - np.array(pop[eachpop-1])
            weight3 = np.array(global_x) - np.array(pop[eachpop-1])
            V[eachpop-1] = list( wV*weight1 + c1*random.random()*weight2 + c2*random.random()*weight3 )

            #判断速度边界
            if V[eachpop-1][0] > Vcmax:
                V[eachpop-1][0] = Vcmax
            if V[eachpop-1][0] < Vcmin:
                V[eachpop-1][0] = Vcmin
            if V[eachpop-1][1] > Vgmax:
                V[eachpop-1][1] = Vgmax
            if V[eachpop-1][1] < Vgmin:
                V[eachpop-1][1] = Vgmin
            if V[eachpop-1][2] > Vhmax:
                V[eachpop-1][2] = Vhmax
            if V[eachpop-1][2] < Vhmin:
                V[eachpop-1][2] = Vhmin

            #种群更新
            pop[eachpop-1] = list( np.array(pop[eachpop-1]) + wP*np.array(V[eachpop-1]) )
            #种群位置边界判断
            if pop[eachpop-1][0] > popcmax:
                pop[eachpop-1][0] = popcmax
            if pop[eachpop-1][0] < popcmin:
                pop[eachpop-1][0] = popcmin
            if pop[eachpop-1][1] > popgmax:
                pop[eachpop-1][1] = popgmax
            if pop[eachpop-1][1] < popgmin:
                pop[eachpop-1][1] = popgmin
            if pop[eachpop-1][2] > pophmax:
                pop[eachpop-1][2] = pophmax
            if pop[eachpop-1][2] < pophmin:
                pop[eachpop-1][2] = pophmin

            #自适应粒子变异                                                        【 matlab代码疑似出错】
            if random.random() > 8:
                k = math.ceil(2*random.random())
                if k == 1:      #【此处有大改动】
                    pop[eachpop-1][k-1] = (popcmax - popcmin)*random.random() + popcmin
                if k == 2:
                    pop[eachpop-1][k-1] = (popgmax - popgmin)*random.random() + popgmin
                if k == 3:
                    pop[eachpop-1][k-1] = (pophmax - pophmin)*random.random() + pophmin
                # if k == 1:      #【原代码】
                #     pop[eachpop-1][k-1] = (popgmax - popgmin)*random.random() + popgmin
                # if k == 2:
                #     pop[eachpop-1][k-1] = (popgmax - popgmin)*random.random() + popgmin
                # if k == 3:
                #     pop[eachpop-1][k-1] = (popgmax - popgmin)*random.random() + popgmin
            
            #求新适应度
            fitness[eachpop-1] = fit_fun(pop[eachpop-1],X,Y,Xt,Yt)

            if outtime_flag == 1:
                exit_flag = True
                break

            
            #个体最优更新
            if fitness[eachpop-1] < local_fitness[eachpop-1]:
                local_x[eachpop-1] = pop[eachpop-1]
                local_fitness[eachpop-1] = fitness[eachpop-1]

            #群体最优更新
            if fitness[eachpop-1] < global_fitness:
                global_x = pop[eachpop-1]
                global_fitness = fitness[eachpop-1]

        if exit_flag == True:
            break

        fit_gen.append(global_fitness)
        avgfitness_gen = sum(fitness)/sizepop

    #输出结果
    print('最优解为:',global_fitness,'\n对应的gam与sig为：',global_x)
    y = global_x.copy()             #  【未用到】
    return global_x,fit_gen


'''
list_T
列表转置

Inputs: 
    inputlist: 输入列表

Output:
    outputlist: 转置后的列表
'''
def list_T(inputlist):
    inputlist = np.array(inputlist)
    inputlist = inputlist.T
    outputlist = inputlist.tolist()
    return outputlist


'''
getcost
获得切换代价
Inputs: 
    costlist:代价表
    before_collocation: 上一时刻的搭配
    mow_collocation：这一时刻的搭配

Output:
    cost: 代价值
'''
def getcost(costlist,before_collocation,now_collocation):
    
    index_list = costlist[:,0].tolist()
    now_collocation_index = index_list.index(now_collocation)
    for eachcollocation_cost in costlist:
        if eachcollocation_cost[0] == before_collocation:
            cost = eachcollocation_cost[now_collocation_index+1]
    return cost


'''
sortorder
返回大小排名
Inputs: 
    sortarray: 待操作数组（一维）
    k: 0 为升序， 1 为降序
   
Output:
    排名数组: 按照原数组顺序，列出原来元素的排名
'''
def sortorder(sortarray,k):
    sortarray_assist = list(set(sortarray))     #去重
    if k == 0:      #排序
        sortarray_assist = sorted(sortarray_assist,reverse=False)
    if k == 1:
        sortarray_assist = sorted(sortarray_assist,reverse=True)

    sortorder_output = [None]*len(sortarray)
    for i in sortarray_assist:
        for j in sortarray:
            if i == j:
                i_index = sortarray_assist.index(i)
                j_index = sortarray.index(j)
                sortarray[j_index] = [None]
                sortorder_output[j_index] = i_index + 1
    return sortorder_output


'''
Inputs: 
    无
   
Output:
    汇总24个时刻泵组推荐前四，至新预测操作假设数据，并打印“汇总完成”
'''
def collection():
    app = xw.App(visible=False,add_book=False)
    wb = app.books.open(r'c:\Users\lz\Desktop\新预测操作.xlsx')
    sht3 = wb.sheets["假设数据"]

    for i in range(1,25):
        sht4 = wb.sheets[str(i)]
        cellname = "G"+str(i+1)
        sht3.range(cellname).options(expand='table').value = sht4.range("D45:D48").value
    wb.save()
    wb.close()
    app.quit()
    print('汇总完毕')


#主函数

#读数据
io = r'c:\Users\lz\Desktop\新提取表格.xls'      #获取文件绝对存储位置
data = pd.read_excel(io, sheet_name =  '1')     #将1工作簿读取到data，data为dataframe类型实例
data = data.values      #将data转化为数组
print('完整数据集为：',data)

#提取搭配表
collocation_24hour_list,collocation_24hour_num = extraction_collocation(data)       #提取每个时刻的泵组搭配及搭配种类数
print('每个小时可采用的搭配：',collocation_24hour_list)
print('每个小时泵组搭配种类数:',collocation_24hour_num)
#print('data类型：',type(data),'collocation类型：',type(collocation_24hour_list))

app = xw.App(visible=False,add_book=False)
wbbeta = app.books.open(r'C:\Users\lz\Desktop\拆分\beta表.xlsx')
shtbaseinfo = wbbeta.sheets["基础数据"]
shtbaseinfo.range('A1').options(expand='table').value = collocation_24hour_num
shtbaseinfo.range('A2').options(expand='table').value = collocation_24hour_list
wbbeta.close()


for timeset in range(1,25):      #暂时设置为一个时间    range(1,2)
    all_collocation_output = []
    count = 0
    
    wbbeta = app.books.open(r'C:\Users\lz\Desktop\拆分\beta表.xlsx')
    shttime = wbbeta.sheets[str(timeset)]
    #提取特定时刻、特定搭配的数据，以待训练
    for collocation_now in collocation_24hour_list[timeset-1]:
       
        
        outtime_count = 0
        outtime_flag = 0
        #collocation_now = 46        #临时监测点
        # if collocation_now == 23:       #指定一两个搭配调试用
        #    continue
    
        if collocation_now == 0:
            break
        print('本次预测泵组搭配为：',collocation_now)
        X,Y,interrupt = dataselect(data,timeset,collocation_now)        
        if interrupt == 1:
            continue
        #printlist(X)
        #printlist(Y)
        X_r = len(X)
        X_c = len(X[0])
        print('行数（数据的条数）：',len(X),'\n','X的列数（输入维度）：',len(X[0]),'\n','Y的列数（输出维度）:',len(Y[0]),'\n')

        #划分数据集
        train_X_raw,test_X_raw,train_Y_raw,test_Y_raw = train_test_split(X,Y,test_size=0.4)      #本次切割为随机切割，【区别于matlab中固定划分】
        
        #对X、Y归一化
        scaler = MinMaxScaler() #实例化
        train_X = scaler.fit_transform(train_X_raw) 
        train_Y = scaler.fit_transform(train_Y_raw)
        test_X = scaler.fit_transform(test_X_raw)
        test_Y = scaler.fit_transform(test_Y_raw)
        print('训练数据X：\n',train_X_raw,'\n训练数据Y：\n',train_Y_raw,'\n测试数据X:\n',test_X_raw,'\n测试数据Y:\n',test_Y_raw)                

        train_X_r = len(train_X)
        train_X_c = len(train_X[0])
        train_Y_c = len(train_Y[0])
        test_X_r = len(test_X)
        test_X_c = len(test_X[0])
        test_Y_c = len(test_Y[0])
        

        #训练模型
        y,trace = psoformsvr(train_X,train_Y,test_X,test_Y)     #参数寻优
        if outtime_flag == 1:
            continue

        C = y[0]
        par = y[1]
        ker = 'rbf'
        tol = 1e-20
        epsi = y[2]

        beta = msvr(train_X,train_Y,'rbf',C,epsi,par,tol)
        Ktest = kernelmatrix(ker,test_X,train_X,par)
        Ypredtest = np.dot(Ktest,beta)
        #print('预测值（归一化）：\n',Ypredtest)

        print(beta)
        
        #反归一化
        test_Y_uniform = scaler.fit_transform(test_Y_raw)       #为下方反归一准备形式 【考虑是否需要将数据集最大最小放进去，差距小预测值不分散】
        Ypredtest = scaler.inverse_transform(Ypredtest)
        #print('预测值反归一化：',Ypredtest)


        #对待预测数据归一化
        io2 = r'c:\Users\lz\Desktop\新预测操作.xlsx'      #获取文件绝对存储位置
        data_simulation = pd.read_excel(io2, sheet_name =  '假设数据')     #将1工作簿读取到data，data为dataframe类型实例
        #data_simulation = [9940,0.295,914,2.2225,6056]
        data_simulation = data_simulation.values      #将data转化为数组
        input_simulation = data_simulation[timeset-1,1:6]
        print('待预测假设数据：',input_simulation)
        X2 = copy.deepcopy(X)
        X2.append(input_simulation.tolist())            #用X归一化待预测数据，然后放入测试集经预测
        X2_afterscaler = scaler.fit_transform(X2)
        input_simulation_afterscaler = X2_afterscaler[-1,:]     #提取出最后一行，为假设数据归一化后的结果
        test_X_simulation = copy.deepcopy(test_X)       #复制新的测试集(归一化后)，用于放置归一化后的待预测数据
        test_Y_simulation = copy.deepcopy(test_Y_raw)       #用于反归一化凑形式
        test_X_simulation = test_X_simulation.tolist()       #转换为list
        # test_Y_simulation = test_Y_simulation.tolist()

        test_X_simulation.append(input_simulation_afterscaler)          #将归一后的待预测数据放入测试集
        Ktest2 = kernelmatrix(ker,test_X_simulation,train_X,par)
        predict_simulation = np.dot(Ktest2,beta)            #用原beta*新Ktest2

        test_Y_simulation.append(test_Y_simulation[-1])       #补齐形式，为接下来的反归一化做准备
        test_Y_simulation = scaler.fit_transform(test_Y_simulation)      #归一化
        predict_simulation_list = scaler.inverse_transform(predict_simulation)        #反归一化
        predict_simulation_output = predict_simulation_list[-1]     #提取出预测结果
        print('这是模拟数据的预测结果：',predict_simulation_output,type(predict_simulation_output))

        #计算R2、混水量/电量与单耗差值
        predict_simulation_list_calculateR2 = predict_simulation_list[0:len(predict_simulation_list)-1]     #去掉最后一个待预测数据，留下测试集的预测值与真实值对比
        predict_simulation_list_calculateR2_T = list_T(predict_simulation_list_calculateR2)       #将预测和真实转置，方便计算其R2
        test_Y_raw_T = list_T(test_Y_raw)
        
        # R2 = r2_score(test_Y_raw_T,predict_simulation_list_calculateR2_T)
        # print(R2)
        R2_add = 0
        for i in range(6):
            R2_add += r2_score(test_Y_raw_T[i],predict_simulation_list_calculateR2_T[i])
        R2 = R2_add / 6     #整体平均值R2评价准确性  【结果偏低】
        # if R2 < 0:
        #     continue        #取消精确度为负值的搭配

        RMSE_output = []        #计算RMSE的值
        for i in range(6):
            RMSE_output.append(sqrt(mean_squared_error(test_Y_raw_T[i], predict_simulation_list_calculateR2_T[i])))

        #开始写内容
        betainfo = shttime.range("B1:AG4").value
        colloction_index = betainfo[0].index(collocation_now)
        colloction_maxrow_index = betainfo[1].index(max(betainfo[1]))
        row = int(betainfo[1][colloction_index])

        if betainfo[1][colloction_index] == 0:
            if betainfo[1][colloction_maxrow_index] == 0:
                row = 6
            else:
                row = int(betainfo[1][colloction_maxrow_index] + betainfo[3][colloction_maxrow_index] + 3)
                print('测试点：',row)

            #写综合数据
            shttime.range('B'+str(row-1)).options(expand='table').value = ['长度','X的行数','X的列数','训练集X的行数','训练集X的列数','测试集X的行数','测试集X的列数','测试集原Y的行数','测试集原y的列数','beta的行数','beta的列数','C','par','ker','tol','epsi','R2','RMSE','RMSE','RMSE','RMSE','RMSE','RMSE']
            shttime.range('A'+str(row)).value = collocation_now
            shttime.range('B'+str(row)).value = X_r
            shttime.range('C'+str(row)).value = X_r
            shttime.range('D'+str(row)).value = X_c
            shttime.range('E'+str(row)).value = train_X_r
            shttime.range('F'+str(row)).value = train_X_c
            shttime.range('G'+str(row)).value = test_X_r
            shttime.range('H'+str(row)).value = test_X_c
            shttime.range('I'+str(row)).value = test_X_r
            shttime.range('J'+str(row)).value = test_Y_c
            shttime.range('K'+str(row)).value = train_X_r
            shttime.range('L'+str(row)).value = test_Y_c
            shttime.range('M'+str(row)).value = C
            shttime.range('N'+str(row)).value = par
            shttime.range('O'+str(row)).value = ker
            shttime.range('P'+str(row)).value = tol
            shttime.range('Q'+str(row)).value = epsi
            shttime.range('R'+str(row)).value = R2
            shttime.range('S'+str(row)).options(expand='table').value = RMSE_output

            #写数据
            shttime.range('B'+str(row+1)).options(expand='table').value = X
            shttime.range('G'+str(row+1)).options(expand='table').value = train_X
            shttime.range('L'+str(row+1)).options(expand='table').value = test_X
            shttime.range('Q'+str(row+1)).options(expand='table').value = test_Y_raw
            shttime.range('W'+str(row+1)).options(expand='table').value = beta

            #将表头的内容更新
            betainfo[1][colloction_index] = row
            betainfo[2][colloction_index] = R2
            betainfo[3][colloction_index] = X_r
            shttime.range('B1:AG4').options(expand='table').value = None
            shttime.range('B1:AG4').options(expand='table').value = betainfo

        elif ( betainfo[1][colloction_index] != 0 ) and ( betainfo[3][colloction_index] >= X_r):
            if betainfo[2][colloction_index] >= R2:
                continue
            if betainfo[3][colloction_index] > X_r:
                clearzone = 'B'+str(betainfo[1][colloction_index]+1)+':AB'+str(betainfo[1][colloction_index]+betainfo[3][colloction_index])
                shttime.range(clearzone).options(expand='table').value = None
            
            #写综合数据
            shttime.range('B'+str(row-1)).options(expand='table').value = ['长度','X的行数','X的列数','训练集X的行数','训练集X的列数','测试集X的行数','测试集X的列数','测试集原Y的行数','测试集原y的列数','beta的行数','beta的列数','C','par','ker','tol','epsi','R2','RMSE','RMSE','RMSE','RMSE','RMSE','RMSE']
            shttime.range('B'+str(row)).value = X_r
            shttime.range('C'+str(row)).value = X_r
            shttime.range('D'+str(row)).value = X_c
            shttime.range('E'+str(row)).value = train_X_r
            shttime.range('F'+str(row)).value = train_X_c
            shttime.range('G'+str(row)).value = test_X_r
            shttime.range('H'+str(row)).value = test_X_c
            shttime.range('I'+str(row)).value = test_X_r
            shttime.range('J'+str(row)).value = test_Y_c
            shttime.range('K'+str(row)).value = train_X_r
            shttime.range('L'+str(row)).value = test_Y_c
            shttime.range('M'+str(row)).value = C
            shttime.range('N'+str(row)).value = par
            shttime.range('O'+str(row)).value = ker
            shttime.range('P'+str(row)).value = tol
            shttime.range('Q'+str(row)).value = epsi
            shttime.range('R'+str(row)).value = R2
            shttime.range('S'+str(row)).options(expand='table').value = RMSE_output

            #写数据
            shttime.range('B'+str(row+1)).options(expand='table').value = X
            shttime.range('G'+str(row+1)).options(expand='table').value = train_X
            shttime.range('L'+str(row+1)).options(expand='table').value = test_X
            shttime.range('Q'+str(row+1)).options(expand='table').value = test_Y_raw
            shttime.range('W'+str(row+1)).options(expand='table').value = beta

            #将表头的内容更新
            betainfo[1][colloction_index] = row
            betainfo[2][colloction_index] = R2
            betainfo[3][colloction_index] = X_r
            shttime.range('B1:AG4').options(expand='table').value = None
            shttime.range('B1:AG4').options(expand='table').value = betainfo

        else :
            if betainfo[2][colloction_index] >= R2:
                continue
            cha =   X_r - betainfo[3][colloction_index]

            #向下平移cha行
            zone_origin = 'A' + str(betainfo[1][colloction_index]+1) + ':AB' + str(betainfo[1][colloction_maxrow_index]+betainfo[3][colloction_maxrow_index])
            shift_arr = shttime.range(zone_origin).value

            zone_new = 'A'+str(betainfo[1][colloction_index]+1+cha)+':AB'+str(betainfo[1][colloction_maxrow_index]+betainfo[3][colloction_maxrow_index]+cha)
            shttime.range(zone_new).options(expand='table').value = None
            shttime.range(zone_new).options(expand='table').value = shift_arr

            #写综合数据
            shttime.range('B'+str(row-1)).options(expand='table').value = ['长度','X的行数','X的列数','训练集X的行数','训练集X的列数','测试集X的行数','测试集X的列数','测试集原Y的行数','测试集原y的列数','beta的行数','beta的列数','C','par','ker','tol','epsi','R2','RMSE','RMSE','RMSE','RMSE','RMSE','RMSE']
            shttime.range('B'+str(row)).value = X_r
            shttime.range('C'+str(row)).value = X_r
            shttime.range('D'+str(row)).value = X_c
            shttime.range('E'+str(row)).value = train_X_r
            shttime.range('F'+str(row)).value = train_X_c
            shttime.range('G'+str(row)).value = test_X_r
            shttime.range('H'+str(row)).value = test_X_c
            shttime.range('I'+str(row)).value = test_X_r
            shttime.range('J'+str(row)).value = test_Y_c
            shttime.range('K'+str(row)).value = train_X_r
            shttime.range('L'+str(row)).value = test_Y_c
            shttime.range('M'+str(row)).value = C
            shttime.range('N'+str(row)).value = par
            shttime.range('O'+str(row)).value = ker
            shttime.range('P'+str(row)).value = tol
            shttime.range('Q'+str(row)).value = epsi
            shttime.range('R'+str(row)).value = R2
            shttime.range('S'+str(row)).options(expand='table').value = RMSE_output

            #写数据
            shttime.range('B'+str(row+1)).options(expand='table').value = X
            shttime.range('G'+str(row+1)).options(expand='table').value = train_X
            shttime.range('L'+str(row+1)).options(expand='table').value = test_X
            shttime.range('Q'+str(row+1)).options(expand='table').value = test_Y_raw
            shttime.range('W'+str(row+1)).options(expand='table').value = beta

            
            

            #更新更改下方的位置信息
            betainfo[1] = [ i+cha if i>betainfo[1][colloction_index] else i for i in betainfo[1] ]

            #将其余表头的内容更新
            betainfo[1][colloction_index] = row
            betainfo[2][colloction_index] = R2
            betainfo[3][colloction_index] = X_r

            #重新写入
            shttime.range('B1:AG4').options(expand='table').value = None
            shttime.range('B1:AG4').options(expand='table').value = betainfo

        print('搭配',collocation_now,'的beta信息已写入\n\n')
        wbbeta.save()
        

    wbbeta.save()
    wbbeta.close()
    print(timeset,'时刻的beta信息已写入')

app.quit()
print('所有时刻运行完成')
    
    
 

