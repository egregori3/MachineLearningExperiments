import os
import multiprocessing as mp
#import DecisionTreeAnalysis
#import BoostingAnalysis
#import KNNAnalysis
#import SVMAnalysis
#import MLPAnalysis

def fun1():
    os.system('python DecisionTreeAnalysis.py > ..//dt.txt')

def fun2():
    os.system('python BoostingAnalysis.py > ..//ada.txt')

def fun3():
    os.system('python SVMAnalysis.py > ..//svm.txt')

def fun4():
    os.system('python KNNAnalysis.py > ..//knn.txt')

def fun5():
    os.system('python MLPAnalysis.py > ..//mlp.txt')

if __name__ == '__main__':
    print("DT Start")
    p1 = mp.Process(target=fun1)
    print("ADA Start")
    p2 = mp.Process(target=fun2)
    print("SVM Start")
    p3 = mp.Process(target=fun3)
    print("KNN Start")
    p4 = mp.Process(target=fun4)
    print("MLP Start")
    p5 = mp.Process(target=fun5)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    print("Waiting for tasks to complete")

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    print("Complete")