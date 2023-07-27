# coding:UTF-8
import numpy as np
import operator
import copy
import matplotlib.pyplot as plt

MIN_VALUE  = 0.0
MAX_VALUE  = 1.0
AX = 50

# benchmark function
def ackley_function(v):
    a = 20.0
    b = 0.20
    c = 2.0*np.pi
    sum1 = 0.0
    sum2 =0.0
    dimy=float(len(v))
    sum1=np.sum(v**2)
    sum2=np.sum(np.cos(c*v))
    return -a*np.exp(-b*np.sqrt(1.0/dimy*sum1))-np.exp(1.0/dimy*sum2)+a+np.exp(1)  

class DE:
    def __init__(self,DIMENSION=2,GENERATION=100,POPULATION=1,min_value=MIN_VALUE,max_value=MAX_VALUE,F=0.5,CR=1.0):
        """
        DIMENSION:説明変数の次元
        GENERATION:世代数
        POPULATION:個体数
        min_value:探索範囲の最小値
        max_value:        最大値
        F:スケーリングファクタ
        CR:交叉確率
        """
        self.DIMENSION=DIMENSION
        self.GENERATION=GENERATION
        self.POPULATION=POPULATION
        self.min_value=min_value
        self.max_value=max_value
        self.F = F
        self.CR = CR
        self.gane = []
        self.result = []

        #data strage
        self.fitness = []
             
        # 初期個体生成
        for p in range(POPULATION):
            self.gane.append(np.random.uniform(self.min_value,self.max_value,(1,self.DIMENSION))[0])

    def mutation(self):
        for i in range(self.POPULATION):
            p = np.arange(self.POPULATION)
            p = np.delete(p,i)
            np.random.shuffle(p)
            
            # v = xr1 + F(xr2-xr3)
            x = self.gane[i]
            v = self.gane[p[0]] + self.F * (self.gane[p[1]] - self.gane[p[2]])
            u = copy.deepcopy(x)
        
            rd = np.random.randint(self.DIMENSION)
            for j in range(rd,self.DIMENSION):
                if (np.random.rand() < self.CR or j == rd):
                    u[j] = v[j]

            #print(self._evaluate(u),self._evaluate(x))
            # if i == self.POPULATION-1:
            #     exit()
            if (self._evaluate(u) < self._evaluate(x)):
                self.gane[i] = u

    def _evaluate(self,gane):
        return ackley_function(gane)

    def evolution(self):
        
        
        for g in range(self.GENERATION):
            if g%10 == 0:
                print(g+1,"世代目")
                # fig = plt.figure(figsize=(5, 5))
                # ax = fig.add_subplot(1,1,1)
                # plt.xlim(-AX,AX)
                # plt.ylim(-AX,AX)
            # for _ in range(len(self.gane)):
            #     ax.scatter(self.gane[_][0],self.gane[_][1],color="blue",s=1)
            #plt.savefig("./plt_de/de_"+str(g)+".png")
            self.mutation()
    
            ans = list(map(self._evaluate, self.gane))
            opt = min(ans)
            self.result.append(opt)

        print("finish")
        return self.result
        # return self.gene[ans.index(min(ans))]