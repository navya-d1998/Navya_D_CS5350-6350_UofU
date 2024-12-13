#!/bin/sh

echo "Training and Test errors for back propogation on bank-note dataset"
python backPropogation.py

echo "Results for stochastic gradient"
python Sgradient.py

echo "Results for stochastic gradient with zero initial weights"
python SgradientZero.py

echo "Results for 2e"
python pytorch.py
