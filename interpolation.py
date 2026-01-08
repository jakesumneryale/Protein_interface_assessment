#!/usr/bin/env python

import os
import argparse
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-u",  help = "unbound pdb", default = "1acb")
parser.add_argument("-b",  help = "bound pdb", default = "1acb")
parser.add_argument("-i",  help = "number of images", type = int, default = 100) #100
parser.add_argument("-d", help='Directory ending in /')

args = parser.parse_args()

input1 = str(args.b)
input2 = str(args.u)

nam = input2.split('.')[-2]
nam = nam.split('/')[-1]

direc = str(args.d) + nam

os.mkdir(direc)
os.mkdir(direc + '/images')

numImages = int(args.i)
coords1 = []
count = 0

with open(input1,'r') as file:
    lines = file.readlines()
    with open(direc + '/images' + '/blank.pdb','a') as blank:
        for l in lines:
            if l[0:4] == 'ATOM':
                if not l[12] == 'H':
                    if not l[13] == 'H':
                        count += 1
                        coords1.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
                        blank.write(l[0:30] + '                        ' + l[54:-1] + '\n')

coords1 = np.array(coords1)

coords2 = []

with open(input2,'r') as file:
    lines = file.readlines()
    for l in lines:
        if l[0:4] == 'ATOM':
            if not l[12] == 'H':
                    if not l[13] == 'H':
                        coords2.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])

coords2 = np.array(coords2)

A = np.transpose(coords1)
B = np.transpose(coords2)
A_centroid = np.average(A, axis=1)[:,None]
B_centroid = np.average(B, axis=1)[:,None] 
H = np.dot((A - A_centroid), np.transpose(B - B_centroid)) 
U, S, Vt = np.linalg.svd(H) 
V = np.transpose(Vt) 
R = np.dot(V, np.transpose(U)) 
#check reflection case
if np.linalg.det(R) < 0:
    V[:,2] *= -1
    R = V * np.transpose(U)
t = B_centroid - np.dot(R, A_centroid)
Atrans = np.dot(R, A) + t
coords1 = np.transpose(Atrans)

fullDelta = coords2-coords1
stepDelta = fullDelta/numImages

for i in range(numImages+1):
    coords = coords1 + (stepDelta * i)
    with open(direc + '/images' + '/' + str(i) + '.pdb','a') as f:
        with open(direc + '/images' + '/blank.pdb','r') as blank:
            lines = blank.readlines()
            for j in range(count):
                x = str(round(coords[j,0],3))
                while(len(x) < 8):
                    x = ' ' + x
                y = str(round(coords[j,1],3))
                while(len(y) < 8):
                    y = ' ' + y
                z = str(round(coords[j,2],3))
                while(len(z) < 8):
                    z = ' ' + z
                f.write(lines[j][0:30] + x + y + z + lines[j][54:-1] + '\n')

os.remove(direc + '/images/blank.pdb')

