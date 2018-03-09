# MLCS7641 Spring 2018 Assignment 2 - Eric Gregori

## Directory Tree
**ABAGAIL-master** - Optimization portion of assignment
**aima-python-master** - Neural network portion of assignment (RHC, SA, GA algorithms in Python)
**DataSets** - Datasets from assignment 1
**emg** - Custom neural network code written in Python
**Experiments** - Just code I had some fun with
**Assignment2_NN.py** - Run this for the neural network portion of assignment

## Optimization problems
### Tools
http://download.oracle.com/otn-pub/java/jdk/8u162-b12/0da788060d494f5095bf8624735fa2f1/jdk-8u162-windows-x64.exe
http://ant.apache.org/bindownload.cgi
https://www.mkyong.com/ant/how-to-install-apache-ant-on-windows/

### Libraries
These problems were addresses using the ABAGAIL library: https://github.com/pushkar/ABAGAIL
The ABAGAIL library was instrumented for data collection.

#### ABAGAIL Customization
All files that start with 'emg' are custom. The following files were modified.
egregori3\ABAGAIL-master\src\opt\ga\StandardGeneticAlgorithm.java
egregori3\ABAGAIL-master\src\opt\prob\MIMIC.java

### Build/Execute Intructions
To build the library and simulations use ant.
See lines below for how to execute each simulation.
The graphs are made by redirecting the simulation output to a file, and importing the file into Excel.
```
egregori3\ABAGAIL-master>ant
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.FourPeaksTest
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgFourPeaksRHC
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgFourPeaksSA
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgFourPeaksGA
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgFourPeaksMIMIC

egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.TwoColorsTest
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgTwoColorsRHC
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgTwoColorsSA
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgTwoColorsGA
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgTwoColorsMIMIC

egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.NQueensTest
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgNQueensRHC
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgNQueensSA
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgNQueensGA
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgNQueensMIMIC

egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgNQueensSAtune
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgNQueensGAtune
egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgNQueensMIMICtune

```

### Example Results
```
D:\GT\ML_CS7641\GIT\Ass2\egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgTwoColorsRHC
20,2,18,1000,0.0,60.719,304.0
40,2,38,1000,33.0,162.17,715.0
80,2,78,997,145.0,403.6228686058175,936.0
160,2,158,585,430.0,787.651282051282,998.0

D:\GT\ML_CS7641\GIT\Ass2\egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgTwoColorsSA
20,2,18,1000,0.0,107.671,283.0
40,2,38,1000,85.0,219.948,624.0
80,2,78,995,205.0,469.851256281407,990.0
160,2,158,456,514.0,827.9714912280701,999.0

D:\GT\ML_CS7641\GIT\Ass2\egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgTwoColorsGA
20,2,18,709,0.0,10.097320169252468,32.0
40,2,38,336,9.0,38.26488095238095,68.0
80,2,78,59,73.0,108.42372881355932,138.0
160,2,158,0,1000.0,NaN,0.0

D:\GT\ML_CS7641\GIT\Ass2\egregori3\ABAGAIL-master>java -cp ABAGAIL.jar opt.test.emgTwoColorsMIMIC
20,2,18,1000,0.0,9.267,47.0
40,2,38,1000,3.0,22.558,116.0
80,2,78,1000,8.0,36.163,134.0
160,2,158,1000,20.0,84.232,245.0

```

---
---
---

## Neural Network Training
### Tools
Python 3
attrs==17.4.0
certifi==2018.1.18
colorama==0.3.9
cycler==0.10.0
matplotlib==2.1.2
numpy==1.14.0
pandas==0.22.0
pluggy==0.6.0
py==1.5.2
pyparsing==2.2.0
pytest==3.3.2
python-dateutil==2.6.1
pytz==2017.3
scikit-learn==0.19.1
scipy==1.0.0
six==1.11.0
tornado==4.5.3
wincertstore==0.2

### Base Code Source
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
http://aima.cs.berkeley.edu/python/search.html
https://github.com/aimacode/aima-python

#### Custom Code
Custom Neural network and evaluation software was created for this project in Python.
This code is in the emg directory.

### Build/Execute Intructions
```
python Assignment2_NN.py
```









* Save your data at real time
* Drag and drop a file to load it
* Ctrl/Cmd + S to save the source file
* Support Github Flavored Markdown syntax
* Support many languages highlight in editor and preview mode

## Syntax Highlight

```javascript
function myFunc(theObject) {
  theObject = {make: "Ford", model: "Focus", year: 2006};
}

var mycar = {make: "Honda", model: "Accord", year: 1998},
    x,
    y;

x = mycar.make;     // x gets the value "Honda"

myFunc(mycar);
y = mycar.make;     // y still gets the value "Honda"
```
