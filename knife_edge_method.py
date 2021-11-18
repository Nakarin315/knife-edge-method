"""
This is a  script for estimating of a beam radius using knife edge method
"""
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special
from sklearn.metrics import r2_score
# Take data from csv file (excel)
data1 = pd.read_csv('vertical_beam.csv')
# take value from CSV file
x=data1['Position (mm)'].values
y=data1['Power (mW)'].values
# difine function for curve fitting
def funct_1(x,x0,p0,p_max,w):
    return p0+(p_max/2)*(1-special.erf(math.sqrt(2)*(x0-x)/w))
# initial guesses for parameters
c0=[7,0,0.342,0.5]
# fit curve with function 
c,cov = curve_fit(funct_1,x,y,c0)
#define the fitting function
yp=funct_1(x,c[0],c[1],c[2],c[3])
print('x0 = %.2f mm'% (c[0]))
print('Background light =  %.3f mW'% (c[1]))
print('Maximum power =  %.3f mW'% (c[2]))
print('Beam waist =  %.2f mm'% (c[3]))
# find R^2 value
print('R^2 : %.5f'%(r2_score(y,yp)))
plt.figure()
plt.title('The measurement of a transverse profile of laser beam by Knife Edge method')
plt.xlabel('Position (mm)')
plt.ylabel('Optical power (mW)')
#Plot data
plt.scatter(x,y,alpha=0.5)
#Plot the fitting function
plt.plot(x,yp)