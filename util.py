from scipy.special import comb
from itertools import combinations
import numpy as np
import copy
import math

def uniformpoint(N, M):
    """
    生成均匀分布的参考点，用于NSGA-III算法中的选择操作
    
    参数:
        N: 期望生成的参考点数量
        M: 目标函数的维度
        
    返回:
        W: 生成的参考点集合，形状为(实际点数, M)
        N: 实际生成的参考点数量
    """
    # 确定参数H1，它控制每个维度上的划分数
    # 增加H1直到能生成足够多的点
    H1 = 1
    while (comb(H1+M-1, M-1) <= N):
        H1 = H1+1
    H1 = H1-1
    
    # 生成初始参考点集
    # 利用组合数学生成单形上均匀分布的点
    W = np.array(list(combinations(range(H1+M-1), M-1))) - np.tile(np.array(list(range(M-1))), (int(comb(H1+M-1, M-1)), 1))
    
    # 计算权重向量，将组合索引转换为实际权重值
    # 每个参考点的坐标和为1，形成单形空间中的均匀分布
    W = (np.hstack((W, H1+np.zeros((W.shape[0], 1)))) - np.hstack((np.zeros((W.shape[0], 1)), W))) / H1
    
    # 当H1<M时，简单均匀划分可能生成的点不够
    # 需要添加一组额外的"内部"参考点
    if H1 < M:
        H2 = 0
        while(comb(H1+M-1, M-1) + comb(H2+M-1, M-1) <= N):
            H2 = H2+1
        H2 = H2-1
        
        if H2 > 0:
            # 生成第二组参考点
            W2 = np.array(list(combinations(range(H2+M-1), M-1))) - np.tile(np.array(list(range(M-1))), (int(comb(H2+M-1, M-1)), 1))
            W2 = (np.hstack((W2, H2+np.zeros((W2.shape[0], 1)))) - np.hstack((np.zeros((W2.shape[0], 1)), W2))) / H2
            
            # 将第二组点向中心偏移
            W2 = W2/2 + 1/(2*M)
            
            # 合并两组参考点
            W = np.vstack((W, W2))
    
    # 避免数值不稳定，设置最小值限制
    W[W < 1e-6] = 1e-6
    
    # 获取实际生成的参考点数量
    N = W.shape[0]
    
    return W, N

def funfun(M,N,name):
    #种群初始化
    D=M+4#定义自变量个数为目标个数加4
    low=np.zeros((1,D))
    up=np.ones((1,D))
    pop = np.tile(low,(N,1))+(np.tile(up,(N,1))-np.tile(low,(N,1)))*np.random.rand(N,D)
    #pop是(N,D)
    
    #计算PF
    if name=='DTLZ1':
        #g=np.transpose(np.mat(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))))
        g=np.array(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))).reshape(N,1)
        #每一个样本计算1个g值。
        popfun=np.multiply(0.5*np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),pop[:,:(M-1)]))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),1-np.fliplr(pop[:,:(M-1)])))))
        #popfun(N,M)，N个样本，每个样本有M个函数值
        P,nouse = uniformpoint(N,M)
        P=P/2
    elif name=='DTLZ2':
        #g=np.transpose(np.mat(np.sum((pop[:,(M-1):]-0.5)**2,1)))
        g=np.array(np.sum((pop[:,(M-1):]-0.5)**2,1)).reshape(N,1)
        popfun=np.multiply(np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),np.cos(pop[:,:(M-1)]*(np.pi/2))))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),1-np.sin(np.fliplr(pop[:,:(M-1)])*(np.pi/2))))))
        P,nouse = uniformpoint(N,M)        
        #P = P/np.tile(np.transpose(np.mat(np.sqrt(np.sum(P**2,1)))),(1,M))
        P = P/np.tile(np.array(np.sqrt(np.sum(P**2,1))).reshape(P.shape[0],1),(1,M))
    elif name=='DTLZ3':
        g=np.array(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))).reshape(N,1)
        popfun=np.multiply(np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),np.cos(pop[:,:(M-1)]*(np.pi/2))))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),1-np.sin(np.fliplr(pop[:,:(M-1)])*(np.pi/2))))))
        P,nouse = uniformpoint(N,M)        
        #P = P/np.tile(np.transpose(np.mat(np.sqrt(np.sum(P**2,1)))),(1,M))
        P = P/np.tile(np.array(np.sqrt(np.sum(P**2,1))).reshape(P.shape[0],1),(1,M))
        
    return pop,popfun,P,D

def cal(pop,name,M,D):
    N = pop.shape[0]
    if name=='DTLZ1':
        g=np.array(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))).reshape(N,1)
        popfun=np.multiply(0.5*np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),pop[:,:(M-1)]))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),1-np.fliplr(pop[:,:(M-1)])))))

    elif name=='DTLZ2':
        g=np.array(np.sum((pop[:,(M-1):]-0.5)**2,1)).reshape(N,1)
        popfun=np.multiply(np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),np.cos(pop[:,:(M-1)]*(np.pi/2))))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),1-np.sin(np.fliplr(pop[:,:(M-1)])*(np.pi/2))))))

    elif name=='DTLZ3':
        g=np.array(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))).reshape(N,1)
        popfun=np.multiply(np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),np.cos(pop[:,:(M-1)]*(np.pi/2))))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),1-np.sin(np.fliplr(pop[:,:(M-1)])*(np.pi/2))))))
    return popfun

def GO(pop,t1,t2,pc,pm):
    pop1 = copy.deepcopy(pop[0:int(pop.shape[0]/2),:])
    pop2 = copy.deepcopy(pop[(int(pop.shape[0]/2)):(int(pop.shape[0]/2)*2),:])
    N,D = pop1.shape[0],pop1.shape[1]
    #模拟二进制交叉
    beta=np.zeros((N,D))
    mu=np.random.random_sample([N,D])
    beta[mu<=0.5]=(2*mu[mu<=0.5])**(1/(t1+1))
    beta[mu>0.5]=(2-2*mu[mu>0.5])**(-1/(t1+1))
    beta=beta*((-1)**(np.random.randint(2, size=(N,D))))
    beta[np.random.random_sample([N,D])<0.5]=1
    beta[np.tile(np.random.random_sample([N,1])>pc,(1,D))]=1
    off = np.vstack(((pop1+pop2)/2+beta*(pop1-pop2)/2,(pop1+pop2)/2-beta*(pop1-pop2)/2))
    #多项式变异
    low=np.zeros((2*N,D))
    up=np.ones((2*N,D))
    site=np.random.random_sample([2*N,D]) < pm
    mu = np.random.random_sample([2*N,D])
    temp = site & (mu<=0.5)
    off[off<low]=low[off<low]
    off[off>up]=up[off>up]
    off[temp]=off[temp]+(up[temp]-low[temp])*((2*mu[temp]+(1-2*mu[temp])*((1-(off[temp]-low[temp])/(up[temp]-low[temp]))**(t2+1)))**(1/(t2+1))-1)
    temp = site & (mu>0.5)
    off[temp]=off[temp]+(up[temp]-low[temp])*(1-(2*(1-mu[temp])+2*(mu[temp]-0.5)*((1-(up[temp]-off[temp])/(up[temp]-low[temp]))**(t2+1)))**(1/(t2+1)))
    
    return off

def NDsort(mixpop,N,M):
    nsort = N#排序个数
    N,M = mixpop.shape[0],mixpop.shape[1]
    Loc1=np.lexsort(mixpop[:,::-1].T)#loc1为新矩阵元素在旧矩阵中的位置，从第一列依次进行排序
    mixpop2=mixpop[Loc1]
    Loc2=Loc1.argsort()#loc2为旧矩阵元素在新矩阵中的位置
    frontno=np.ones(N)*(np.inf)#初始化所有等级为np.inf
    #frontno[0]=1#第一个元素一定是非支配的
    maxfno=0#最高等级初始化为0
    while (np.sum(frontno < np.inf) < min(nsort,N)):#被赋予等级的个体数目不超过要排序的个体数目
        maxfno=maxfno+1
        for i in range(N):
            if (frontno[i] == np.inf):
                dominated = 0
                for j in range(i):
                    if (frontno[j] == maxfno):
                        m=0
                        flag=0
                        while (m<M and mixpop2[i,m]>=mixpop2[j,m]):
                            if(mixpop2[i,m]==mixpop2[j,m]):#相同的个体不构成支配关系
                                flag=flag+1
                            m=m+1 
                        if (m>=M and flag < M):
                            dominated = 1
                            break
                if dominated == 0:
                    frontno[i] = maxfno
    frontno=frontno[Loc2]
    return frontno,maxfno
#求两个向量矩阵的余弦值,x的列数等于y的列数
def pdist(x,y):
    x0=x.shape[0]
    y0=y.shape[0]
    xmy=np.dot(x,y.T)#x乘以y
    xm=np.array(np.sqrt(np.sum(x**2,1))).reshape(x0,1)
    ym=np.array(np.sqrt(np.sum(y**2,1))).reshape(1,y0)
    xmmym=np.dot(xm,ym)
    cos = xmy/xmmym
    return cos

def lastselection(popfun1,popfun2,K,Z,Zmin):
    #选择最后一个front的解
    popfun = copy.deepcopy(np.vstack((popfun1, popfun2)))-np.tile(Zmin,(popfun1.shape[0]+popfun2.shape[0],1))
    N,M = popfun.shape[0],popfun.shape[1]
    N1 = popfun1.shape[0]
    N2 = popfun2.shape[0]
    NZ = Z.shape[0]
    
    #正则化
    extreme = np.zeros(M)
    w = np.zeros((M,M))+1e-6+np.eye(M)
    for i in range(M):
        extreme[i] = np.argmin(np.max(popfun/(np.tile(w[i,:],(N,1))),1))
    
    #计算截距
    extreme = extreme.astype(int)#python中数据类型转换一定要用astype
    #temp = np.mat(popfun[extreme,:]).I
    temp = np.linalg.pinv(np.asmatrix(popfun[extreme,:]))
    hyprtplane = np.array(np.dot(temp,np.ones((M,1))))
    a = 1/hyprtplane
    if np.sum(a==math.nan) != 0:
        a = np.max(popfun,0)
    np.array(a).reshape(M,1)#一维数组转二维数组
    #a = a.T - Zmin
    a=a.T
    popfun = popfun/(np.tile(a,(N,1)))
    
    ##联系每一个解和对应向量
    #计算每一个解最近的参考线的距离
    cos = pdist(popfun,Z)
    distance = np.tile(np.array(np.sqrt(np.sum(popfun**2,1))).reshape(N,1),(1,NZ))*np.sqrt(1-cos**2)
    #联系每一个解和对应的向量
    d = np.min(distance.T,0)
    pi = np.argmin(distance.T,0)
    
    #计算z关联的个数
    rho = np.zeros(NZ)
    for i in range(NZ):
        rho[i] = np.sum(pi[:N1] == i)
    
    #选出剩余的K个
    choose = np.zeros(N2)
    choose = choose.astype(bool)
    zchoose = np.ones(NZ)
    zchoose = zchoose.astype(bool)
    while np.sum(choose) < K:
        #选择最不拥挤的参考点
        temp = np.ravel(np.array(np.where(zchoose == True)))
        jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))
        j = temp[jmin[np.random.randint(jmin.shape[0])]]
#        I = np.ravel(np.array(np.where(choose == False)))
#        I = np.ravel(np.array(np.where(pi[(I+N1)] == j)))
        I = np.ravel(np.array(np.where(pi[N1:] == j)))
        I = I[choose[I] == False]
        if (I.shape[0] != 0):
            if (rho[j] == 0):
                s = np.argmin(d[N1+I])
            else:
                s = np.random.randint(I.shape[0])
            choose[I[s]] = True
            rho[j] = rho[j]+1
        else:
            zchoose[j] = False
    return choose

def lastselection_scheduling(popfun1, popfun2, K, Z, Zmin):
    """针对调度问题的最后前沿选择"""
    popfun = np.vstack((popfun1, popfun2)) - np.tile(Zmin, (popfun1.shape[0] + popfun2.shape[0], 1))
    N, M = popfun.shape[0], popfun.shape[1]
    N1 = popfun1.shape[0]
    N2 = popfun2.shape[0]
    NZ = Z.shape[0]
    
    # 标准化
    if np.max(popfun) > 0:
        a = np.max(popfun, axis=0)
        a[a == 0] = 1
        popfun = popfun / np.tile(a, (N, 1))
    
    # 计算每个解到参考线的距离
    cos = pdist(popfun, Z)
    distance = np.tile(np.sqrt(np.sum(popfun**2, 1)).reshape(N, 1), (1, NZ)) * np.sqrt(1 - cos**2)
    
    # 关联每个解到最近的参考线
    d = np.min(distance.T, 0)
    pi = np.argmin(distance.T, 0)
    
    # 计算每条参考线关联的解的数量
    rho = np.zeros(NZ)
    for i in range(NZ):
        rho[i] = np.sum(pi[:N1] == i)
    
    # 选择剩余的K个解
    choose = np.zeros(N2, dtype=bool)
    zchoose = np.ones(NZ, dtype=bool)
    
    while np.sum(choose) < K:
        # 选择最不拥挤的参考线
        temp = np.where(zchoose == True)[0]
        jmin = np.where(rho[temp] == np.min(rho[temp]))[0]
        j = temp[jmin[np.random.randint(len(jmin))]]
        
        # 找到关联到参考线j的候选解
        I = np.where(pi[N1:] == j)[0]
        I = I[choose[I] == False]
        
        if len(I) != 0:
            if rho[j] == 0:
                s = np.argmin(d[N1 + I])
            else:
                s = np.random.randint(len(I))
            choose[I[s]] = True
            rho[j] = rho[j] + 1
        else:
            zchoose[j] = False
    
    return choose

def envselect(mixpop,N,Z,Zmin,name,M,D):
    #非支配排序
    mixpopfun = cal(mixpop,name,M,D)
    frontno,maxfno = NDsort(mixpopfun,N,M)
    Next = frontno < maxfno
    #选择最后一个front的解
    Last = np.ravel(np.array(np.where(frontno == maxfno)))
    choose = lastselection(mixpopfun[Next,:],mixpopfun[Last,:],N-np.sum(Next),Z,Zmin)
    Next[Last[choose]] = True
    #生成下一代
    pop = copy.deepcopy(mixpop[Next,:])
    return pop

def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)
 
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0   
    ED = np.sqrt(SqED)
    return ED

def IGD(popfun,PF):
    distance = np.min(EuclideanDistances(PF,popfun),1)
    score = np.mean(distance)
    return score