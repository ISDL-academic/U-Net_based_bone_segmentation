import numpy as np
from DE import DE
from jDE import jDE
import sys
import matplotlib.pyplot as plt

MIN_VALUE  = -32
MAX_VALUE  = 32

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

def excute_DE():
    print("DE")
    D = 10
    #print(ackley_function(np.array([0 for i in range(D)])))
    # print(ackley_function(np.array([1,1])))
    de = DE(DIMENSION=D,POPULATION=100,GENERATION=100,min_value=MIN_VALUE,max_value=MAX_VALUE)
    opt = de.evolution()

    fig = plt.figure(figsize=(5, 5))
    plt.plot(opt)
    plt.show()
    #ax = fig.add_subplot(1,1,1)
    #print(ackley_function(np.array(opt)))

def excute_jDE(G_num=10,P_num=100):
    print("jDE")
    jde = jDE(GENERATION=G_num,POPULATION=P_num)
    jde.evolution(monitor=False)


if __name__ == "__main__":
    #excute_DE()
    excute_jDE(10,100)
    # print("jDE")
    # jde = jDE(GENERATION=100000)
    # jde.evolution(monitor=True)