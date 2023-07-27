# coding:UTF-8
import numpy as np
import operator
import copy
import matplotlib.pyplot as plt
import iou
import pydicom
from skimage import io
import windowing as wd
from skimage.transform import rescale,resize

MIN_VALUE  = 0
MAX_VALUE  = 1500
AX = 1


# 最適化に使う画像データのパス
# Spine 
inputdir = ["../dcm_spine/1-125.dcm","../dcm_spine/1-180.dcm","../dcm_spine/1-235.dcm","../dcm_spine/1-290.dcm","../dcm_spine/1-345.dcm"]
GTdir = ["../GT/1-125.png","../GT/1-180.png","../GT/1-235.png","../GT/1-290.png","../GT/1-345.png"]

# Femur
# inputdir = ["../dcm_femur/1-440.dcm","../dcm_femur/1-480.dcm","../dcm_femur/1-520.dcm","../dcm_femur/1-560.dcm","../dcm_femur/1-600.dcm",]
# GTdir = ["../GT_femur/1-440.png","../GT_femur/1-480.png","../GT_femur/1-520.png","../GT_femur/1-560.png","../GT_femur/1-600.png"]

# Artificail
# inputdir = ["../dcm_artificial/IMG26.dcm","../dcm_artificial/IMG48.dcm","../dcm_artificial/IMG70.dcm","../dcm_artificial/IMG90.dcm","../dcm_artificial/IMG110.dcm",]
# GTdir = ["../GT_artificial/IMG26.png","../GT_artificial/IMG48.png","../GT_artificial/IMG70.png","../GT_artificial/IMG90.png","../GT_artificial/IMG110.png"]

N = 256
n = len(inputdir)

class Individual:
    def __init__(self,gene = None, min_value=MIN_VALUE,max_value=MAX_VALUE,DIMENSION=2):
        """
        gene:個体
        max_value:ww,wcの最大値
        min_value:　　　　最小値
        DIMENSION:次元
        F:スケーリングファクタ
        CR:交叉確率
        fitness:フィットネス値
        """
        self.DIMENSION=DIMENSION
        if gene is None:
            self.gene = np.random.uniform(min_value,max_value,(1,self.DIMENSION))[0]
        else:
            self.gene = gene
        
        self.fitness = self.evaluate(self.gene)
        self.F  = 0.5
        self.CR = 0.9
    
    
    def evaluate(self,gene):
        #print("評価")
        current_iou = 0
        for i in range(n):
          ds = pydicom.dcmread(inputdir[i])
          dcm = ds.pixel_array.astype(float)
          tmp = wd.windowing(dcm,gene[0],gene[1])
          img = resize(tmp[0],(256,256),mode='reflect',anti_aliasing=True)
          img_GT = io.imread(GTdir[i])
          if img_GT.shape==(256,256,4):
            img_GT = np.delete(img_GT, 3, 2) # αチャンネルがあれば削除
          
          for i in range(N):
              for j in range(N):
                  #print(img[i,j,:]*255)
                  if list(img[i,j,:]*255) > [200,200,200]:
                      img[i,j,0] = 255
                      img[i,j,1] = 255
                      img[i,j,2] = 0
                  else :
                      img[i,j,0] = 0
                      img[i,j,1] = 0
                      img[i,j,2] = 0
          # print(img)
          # print(img_GT)
          current_iou += iou.calcIoU(img,img_GT)
        current_iou /= n
        #print(current_iou,iou.calcIoU(img,img_GT))
        return 1 - current_iou
        #return ackley_function(gene)
        
class jDE:
    def __init__(self,DIMENSION=2,GENERATION=10,POPULATION=100,F_l=0.1,F_u=0.9,gamma1=0.1,gamma2=0.2):
        """
        DIMENSION:説明変数の次元
        GENERATION:世代数
        POPULATION:個体数
        F_l 
        F_u
        gamma1
        gamma2
        """
        self.DIMENSION=DIMENSION
        self.GENERATION=GENERATION
        self.POPULATION=POPULATION
        self.Population = []
        # mutation parameters
        self.F_l=F_l # Fの下限値
        self.F_u=F_u # 上限値
        self.gamma1=gamma1
        self.gamma2=gamma2
        
        #data strage
        self.history_fitness = []
        self.history_CR = []
        self.history_F = []        
        
        print("世代数",GENERATION,"個体数",POPULATION)
        for p in range(POPULATION):
            # 初期個体
            fixed_gene = np.random.uniform(MIN_VALUE,MAX_VALUE,(1,self.DIMENSION))[0]
            self.Population.append(Individual(fixed_gene))
            self.history_CR.append([])
            self.history_F.append([])      
              
    def mutation(self):
        for i in range(self.POPULATION):
            p = np.arange(self.POPULATION)
            p = np.delete(p,i)
            np.random.shuffle(p)

            # jDE Algorithm. Decide parameters F and CR
            if (np.random.rand() < self.gamma1):
                self.Population[i].F  = self.F_l + np.random.rand()*self.F_u
            
            if (np.random.rand() < self.gamma2):
                self.Population[i].CR = np.random.rand() 
            
            
            v = self.Population[p[0]].gene + self.Population[i].F * (self.Population[p[1]].gene - self.Population[p[2]].gene)
            u = v
            #for gene in self.Population:
            # for i in range(len(self.Population)):
            #     print(self.Population[i].gene)
            for j in range(np.random.randint(self.DIMENSION-1),self.DIMENSION):
                if (np.random.rand() >  1.0-self.Population[i].CR):
                    u[j] = self.Population[i].gene[j]
            
            # 新しく生成された子個体のフィットネス値を計算
            new_Individual = Individual()
            new_Individual.fitness = new_Individual.evaluate(u)

            #Set new parameters F and CR            
            new_Individual.F = self.Population[i].F
            new_Individual.CR = self.Population[i].CR
            
            if (new_Individual.fitness < self.Population[i].fitness):
                self.Population[i] = new_Individual

            self.history_CR[i].append(self.Population[i].CR)
            self.history_F[i].append(self.Population[i].F)

    def average_fitness(self):        
        x =0.0
        for i in range(self.POPULATION):
            x += self.Population[i-1].fitness
        return x/float(self.POPULATION)
        
    def evolution(self,monitor=True):
        for g in range(self.GENERATION):
            #if g%500 == 0:
            print(g+1,"世代目")
            self.mutation()
            self.Population.sort(key = operator.attrgetter('fitness'),reverse = False)
            if monitor and g%2000 == 0:
                self.plot_fitness()
                self.plot_F()
                self.plot_CR()
            self.print_best_individual()
            self.history_fitness.append(self.Population[0].fitness)
        print("finish")
        
    def plot_fitness(self):
        plt.plot(self.history_fitness)
        plt.plot(self.history_fitness,'-o')
        plt.xlabel("Generation",size = 16)
        plt.ylabel("Fitness",size = 16)
        plt.show()

    def plot_CR(self,indiv_list=[0]):
        for n in indiv_list:
            plt.plot(self.history_CR[n],'-o')
        plt.xlabel("Generation",size = 16)
        plt.ylabel("CR",size = 16)
        plt.show()

    def plot_F(self,indiv_list=[0]):
        for n in indiv_list:
            plt.plot(self.history_F[n],'-o')
        plt.xlabel("Generation",size = 16)
        plt.ylabel("F",size = 16)
        plt.show()         
            
    def print_best_individual(self):
        self.Population.sort(key = operator.attrgetter('fitness'),reverse = False)
        print("print best individual",self.Population[0].fitness,self.Population[0].gene)
        for i in range(3):
          print("print some individuals",self.Population[i].fitness,self.Population[i].gene)