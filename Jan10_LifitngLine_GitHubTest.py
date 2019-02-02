# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:24:08 2019

@author: radha
"""

#from numpy import *
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
import pickle
from makewindVeers_main import *
import seaborn as sns
import scipy.stats as st
import statistics as sts
import scipy.fftpack
import scipy.special
from aP import *
from HermitePoly import myHermite
import scipy.io as sio
from datetime import date
import time
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import

#%%
f = open('Gamma_realizations.pckl', 'rb')
Gamma_all = pickle.load(f)
f.close()
#%%
f = open('WindSpeed_realizations.pckl','rb')
WindSpeed_all = pickle.load(f)
f.close()

f = open('xi_relaizaitons.pckl','rb')
xi_all = pickle.load(f)
f.close()

#%%



ITStep = 60

N = len(Gamma_all)
SimLength = Gamma_all.shape[1]
NoOfRlz = np.array([48000,24000,1200,600,120,60,12,6])
SeedNo = np.empty((NoOfRlz.shape[0],NoOfRlz.max()))
SeedNo[:] = None

GammaMean = np.zeros((NoOfRlz.shape[0],SimLength))
GammaStd = np.zeros((NoOfRlz.shape[0],SimLength))

for i in range(0,NoOfRlz.shape[0]):
    SeedNo[i,0:NoOfRlz[i]] = np.random.randint(0,50000,NoOfRlz[i])

for i in range(0,NoOfRlz.shape[0]):
    print(i)
    for j in range(ITStep,SimLength):
       GammaMean[i,j] = np.mean(Gamma_all[SeedNo[i,0:NoOfRlz[i]].astype(int),j]) 
       GammaStd[i,j] = np.std(Gamma_all[SeedNo[i,0:NoOfRlz[i]].astype(int),j])
#%%
ITStep = 60

N = len(Gamma_all)
SimLength = Gamma_all.shape[1]
NoOfRlz_6 = np.array([6])
SeedNo_6 = np.empty((NoOfRlz_6.shape[0],48000))
SeedNo_6[:] = None

GammaMean_6 = np.zeros((NoOfRlz_6.shape[0],SimLength))
GammaStd_6 = np.zeros((NoOfRlz_6.shape[0],SimLength))

for i in range(0,NoOfRlz_6.shape[0]):
    SeedNo_6[i,0:NoOfRlz_6[i]] = np.random.randint(0,50000,NoOfRlz_6[i])

for i in range(0,NoOfRlz_6.shape[0]):
    print(i)
    for j in range(ITStep,SimLength):
       GammaMean_6[i,j] = np.mean(Gamma_all[SeedNo_6[i,0:NoOfRlz_6[i]].astype(int),j]) 
       GammaStd_6[i,j] = np.std(Gamma_all[SeedNo_6[i,0:NoOfRlz_6[i]].astype(int),j])
#%%
GammaMean = np.vstack((GammaMean,GammaMean_6))
GammaStd = np.vstack((GammaStd,GammaStd_6))
SeedNo = np.vstack((SeedNo,SeedNo_6))
#%%
plt.close('all')  
plt.figure()

  
x = np.arange(ITStep+1,SimLength)

       
for i in range(0,NoOfRlz.shape[0]):
    plt.subplot(7,1,i+1)
    plt.plot(x/10,GammaMean[i,ITStep:-1],label = str(NoOfRlz[i]))
    plt.fill_between(x/10,GammaMean[i,ITStep:-1]-GammaStd[i,ITStep:-1],GammaMean[i,ITStep:-1]+GammaStd[i,ITStep:-1],alpha=0.1)
    plt.legend(loc='upper right')



plt.grid(True)
plt.show()


#%%
polyOrder = 2
InitialDist = cp.Normal()


NoOfSamples=np.array([6,12,24,48,96,192,384])

SimLength = Gamma_all.shape[1]


GammaMeanPCE = np.zeros((NoOfSamples.shape[0],SimLength))
GammaStdPCE = np.zeros((NoOfSamples.shape[0],SimLength))


for i in range(0,NoOfSamples.shape[0]):
    dist = cp.Normal(np.mean(WindSpeed_all[:,j]),np.std(WindSpeed_all[:,j]))
    orthPoly = cp.orth_ttr(polyOrder, dist)
    dataPoints_initial = dist.sample(NoOfSamples[i],rule='L')
    dataPoints = np.zeros(NoOfSamples[i])
    samples_u = np.zeros(NoOfSamples[i])
        #idx = nNoOfSamplesp.zeros(NoOfSamples[i])
    u = Gamma_all[:,j]
    ws= WindSpeed_all[:,j]
        for k in range(0,NoOfSamples[i]):
            idx= (np.abs(ws - dataPoints_initial[k])).argmin()
            dataPoints[k] = ws[idx]
            samples_u[k] = u[idx]
    for j in range(0,SimLength):
        dist = cp.Normal(np.mean(WindSpeed_all[:,j]),np.std(WindSpeed_all[:,j]))
        orthPoly = cp.orth_ttr(polyOrder, dist)
        dataPoints_initial = dist.sample(NoOfSamples[i],rule='L')
        dataPoints = np.zeros(NoOfSamples[i])
        samples_u = np.zeros(NoOfSamples[i])
        #idx = nNoOfSamplesp.zeros(NoOfSamples[i])
        u = Gamma_all[:,j]
        ws= WindSpeed_all[:,j]
        for k in range(0,NoOfSamples[i]):
            idx= (np.abs(ws - dataPoints_initial[k])).argmin()
            dataPoints[k] = ws[idx]
            samples_u[k] = u[idx]
        approx = cp.fit_regression(orthPoly, dataPoints, samples_u)
        GammaMeanPCE[i,j] = cp.E(approx, dist)
        GammaStdPCE[i,j] = cp.Std(approx, dist)
        print(i,j)

#%%
timestr = time.strftime("%Y%m%d")

f = open(timestr+'_GammaMean.pckl', 'wb')
pickle.dump(GammaMean, f)
f.close()

f = open(timestr+'_GammaStd.pckl', 'wb')
pickle.dump(GammaStd, f)
f.close()

f = open(timestr+'_SeedNo.pckl', 'wb')
pickle.dump(SeedNo, f)
f.close()

#%%
                
f = open(timestr+'GammaMeanPCE.pckl', 'wb')
pickle.dump(GammaMeanPCE, f)
f.close()

f = open(timestr+'GammaStdPCE.pckl', 'wb')
pickle.dump(GammaStdPCE, f)
f.close()


f = open(timestr+'NoOfSamplesPCE.pckl', 'wb')
pickle.dump(NoOfSamples, f)
f.close()


#%%

f = open('20190130GammaMeanPCE.pckl', 'rb')
GammaMeanPCE = pickle.load(f)
f.close()

f = open('20190130GammaStdPCE.pckl','rb')
GammaStdPCE = pickle.load(f)
f.close()

f = open('20190201_GammaMean.pckl', 'rb')
GammaMean = pickle.load(f)
f.close()

f = open('20190201_GammaStd.pckl','rb')
GammaStd = pickle.load(f)
f.close()

f = open('20190201_SeedNo.pckl','rb')
SeedNo = pickle.load(f)
f.close()


#%%


plt.close('all')  
plt.figure()

ITStep = 60
SimLength = Gamma_all.shape[1]
NoOfRlz = np.array([48000,24000,1200,600,120,60,12,6])

  
x = np.arange(ITStep+1,SimLength)

k=6
       
for i in range(0,NoOfSamples.shape[0]-4):
    plt.subplot(NoOfSamples.shape[0],1,i+1)
    plt.plot(x/10,GammaMeanPCE[i,ITStep:-1],label = str(NoOfSamples[i]))
    plt.fill_between(x/10,GammaMeanPCE[i,ITStep:-1]-GammaStdPCE[i,ITStep:-1],GammaMeanPCE[i,ITStep:-1]+GammaStdPCE[i,ITStep:-1],alpha=0.1)
    plt.plot(x/10,GammaMean[k,ITStep:-1],label = str(NoOfRlz[k]))
    plt.fill_between(x/10,GammaMean[k,ITStep:-1]-GammaStd[k,ITStep:-1],GammaMean[k,ITStep:-1]+GammaStd[k,ITStep:-1],alpha=0.1)
    plt.legend(loc='upper right')
    plt.grid(True)


plt.figure()

i=0
k=6

bgTst = 100
enTst = 105

plt.plot(x[bgTst:enTst]/10,GammaMeanPCE[i,bgTst:enTst],'o',label = str(NoOfSamples[i]))
plt.fill_between(x[bgTst:enTst]/10,GammaMeanPCE[i,bgTst:enTst]-GammaStdPCE[i,bgTst:enTst],GammaMeanPCE[i,bgTst:enTst]+GammaStdPCE[i,bgTst:enTst],alpha=0.1)
plt.plot(x[bgTst:enTst]/10,GammaMean[k,bgTst:enTst],'o',label = str(NoOfRlz[k]))
plt.fill_between(x[bgTst:enTst]/10,GammaMean[k,bgTst:enTst]-GammaStd[k,bgTst:enTst],GammaMean[k,bgTst:enTst]+GammaStd[k,bgTst:enTst],alpha=0.1)
k=0
plt.plot(x[bgTst:enTst]/10,GammaMean[k,bgTst:enTst],'o',label = str(NoOfRlz[k]))
plt.fill_between(x[bgTst:enTst]/10,GammaMean[k,bgTst:enTst]-GammaStd[k,bgTst:enTst],GammaMean[k,bgTst:enTst]+GammaStd[k,bgTst:enTst],alpha=0.1)
plt.legend(loc='upper right')
plt.grid(True)


plt.show()
#%% For the abstracet
plt.close('all')  

timestr = time.strftime("%Y%m%d")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


bgTst = 2500
enTst = 3500

plt.figure()

NoOfRlzS = np.array([7,6,4])

for i in NoOfRlzS:
    plt.subplot(2,1,1)
    plt.plot(x[bgTst:enTst]/10,GammaMean[i,bgTst:enTst],'o',label=str(NoOfRlz[i])+' sims')
    #plt.xlabel(r'Time [Sec]',fontsize = 16)
    plt.ylabel(r'Circulation ($\Gamma$) mean',fontsize = 14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tick_params(labelbottom=False)
    plt.subplot(2,1,2)
    plt.plot(x[bgTst:enTst]/10,GammaStd[i,bgTst:enTst],'o',label=str(NoOfRlz[i])+' sims')
    plt.xlabel(r'Time [Sec]',fontsize = 14)
    plt.tick_params(labelsize = 10)
    plt.ylabel(r'Circulation ($\Gamma$) std',fontsize = 14)
    plt.grid(True)
    plt.legend(loc='upper right')

pfname_timestep = timestr+'Gamma_std_mean_6_12_120.png'
plt.savefig(pfname_timestep, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype='b05', format="png",
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


#%% For the Abstract

bgTst = 2500
enTst = 3500
k=4

timestr = time.strftime("%Y%m%d")


plt.rc('text', usetex=True)
plt.rc('font', family='serif')



plt.figure()

for i in range(0,NoOfSamples.shape[0]-6):
    plt.subplot(2,1,1)
    plt.plot(x[bgTst:enTst]/10,GammaMeanPCE[i,bgTst:enTst],label=str(NoOfSamples[i])+' sampes PCE',lw = 2.0)
    plt.plot(x[bgTst:enTst]/10,GammaMean[k,bgTst:enTst],'o',label=str(NoOfRlz[k])+' sims Mean')
    plt.plot(x[bgTst:enTst]/10,GammaMean[k+3,bgTst:enTst],'o',label=str(NoOfRlz[k+3])+' sims Mean')
    plt.tick_params(labelbottom=False)
    #plt.xlabel(r'Time [Sec]',fontsize = 16)
    plt.ylabel(r'Circulation ($\Gamma$) mean',fontsize = 14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.subplot(2,1,2)
    plt.plot(x[bgTst:enTst]/10,GammaStdPCE[i,bgTst:enTst],label=str(NoOfSamples[i])+' samples PCE', lw = 2.0)
    plt.plot(x[bgTst:enTst]/10,GammaStd[k,bgTst:enTst],'o',label=str(NoOfRlz[k])+' sims Std')
    plt.plot(x[bgTst:enTst]/10,GammaStd[k+3,bgTst:enTst],'o',label=str(NoOfRlz[k+3])+' sims Std')
    plt.xlabel(r'Time [Sec]',fontsize = 14)
    plt.ylabel(r'Circulation ($\Gamma$) std',fontsize = 14)
    plt.tick_params(labelsize = 10)
    plt.grid(True)
    plt.legend(loc='upper right')


pfname_timestep = timestr+'Gamma_std_mean_6PCE_6_120.png'
plt.savefig(pfname_timestep, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype='b05', format="png",
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)






#%%



ax.plot(x[bgTst:enTst]/10,Gamma_all[3000,bgTst:enTst], zs=0, zdir='z')
ax.plot(x[bgTst:enTst]/10,Gamma_all[4000,bgTst:enTst], zs=0, zdir='z')
ax.plot(x[bgTst:enTst]/10,Gamma_all[5000,bgTst:enTst], zs=0, zdir='z')
ax.plot(x[bgTst:enTst]/10,Gamma_all[42534,bgTst:enTst], zs=0, zdir='z')
ax.plot(x[bgTst:enTst]/10,Gamma_all[25000,bgTst:enTst], zs=0, zdir='z')
ax.plot(x[bgTst:enTst]/10,Gamma_all[32023,bgTst:enTst], zs=0, zdir='z')



ax.legend()
#ax.set_xlim(0, 1)
#ax.set_ylim(-5, 2)
#ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#%%
plt.close('all')  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nbins = 25

i=0
k=6


for z in range(bgTst,enTst):
    ys = np.random.normal(GammaMeanPCE[i,z],GammaStdPCE[i,z],2000)
    xl = np.linspace(ys.min(),ys.max(),2000)
    y_pdf=st.norm.pdf(xl,GammaMeanPCE[i,z],GammaStdPCE[i,z])
    hist, bins = np.histogram(ys,nbins,density='True')
    xs = (bins[:-1] + bins[1:])/2
    ax.bar(xs, hist, zs=x[z]/10, zdir='x', alpha=0.2,width =0.8,color='b' )
    #ax.plot(xl,y_pdf,zs=x[z]/10,zdir = 'x')
    del ys,xl,y_pdf,hist,bins
    ys = np.random.normal(GammaMean[k,z],GammaStd[k,z],2000)
    xl = np.linspace(ys.min(),ys.max(),2000)
    y_pdf=st.norm.pdf(xl,GammaMean[k,z],GammaStd[k,z])
    hist, bins = np.histogram(ys,nbins,density='True')
    xs = (bins[:-1] + bins[1:])/2
    ax.bar(xs, hist, zs=x[z]/10, zdir='x', alpha=0.2,width =0.8, color='r' )
        #ax.fill(xl,y_pdf)
    #ax.fill_between(xl, 0, y_pdf, facecolor='blue')
    #xs = (bins[:-1] + bins[1:])/2
    print(x[z]/10)
    
    #ax.bar(xs, hist, zs=x[z]/10, zdir='x', color='r', ec='r', alpha=0.2)

ax.set_ylim(-5, 20)


plt.show()



ax.view_init(30,45)
