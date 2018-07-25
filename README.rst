DDOSdetector:
########################################

this repository contains a notebook and data for training and testing a ddos detector using machine learning. 

How to use:
=============

clone repository :

	clone git@github.com:Reda-Abdellah/DDOSdetector.git 

before executing the code on the notebook unzip data+ipasn.tar.gz in the repository root folder:

	cd DDOSdetector
	tar -xvzf data+ipasn.tar.gz
 
install required packages:

	pip install numpy pandas sklearn pyasn
	jupyter-notebook 

and then choose ddos_detector.ipynb

if you don't have jupyter-notebook installed you can run the .py :
	python ddos_detector.py 

all these sub-datasets are made from august.week1.csv URG16

we just reduced them with linux terminal for computetional matter :

#sed -n '30000000,32500000 p' august.week1.csv > ddos1range32M.csv

#sed -n '104000000,107000000 p' august.week1.csv > ddos1range105M.csv

#head -n 2100000 ddos1range32M.csv | tail -n 600000 > reduced.csv

#head -n 1700000 ddos2range105M.csv | tail -n 400000 > reduced2.csv

#head -n 2999757 ddos2range105M.csv | tail -n 400000 > reduced3.csv


