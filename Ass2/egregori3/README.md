# MLCS7641 Spring 2018 Assignment 2 - Eric Gregori

## Tools
http://download.oracle.com/otn-pub/java/jdk/8u162-b12/0da788060d494f5095bf8624735fa2f1/jdk-8u162-windows-x64.exe
http://ant.apache.org/bindownload.cgi
https://www.mkyong.com/ant/how-to-install-apache-ant-on-windows/

## This assignment has two parts

### Optimization problems
These problems were addresses using the ABAGAIL library: https://github.com/pushkar/ABAGAIL
The ABAGAIL library was instrumented for data collection.

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
```

#### Results
rhc_fourpeaks.txt = Random Hill Climbing four peaks results


### Neural Network Training



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

## Feedback

https://github.com/chenzhiwei/chrome-markdown-editor