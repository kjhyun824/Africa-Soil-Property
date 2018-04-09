#!/bin/bash

Trains="100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000"

for elem in ${Trains}; do
   echo \[DEBUG\] Num Train : ${elem} Starts
   python mult_logistic.py ${elem}
   echo \[DEBUG\] Done
done
