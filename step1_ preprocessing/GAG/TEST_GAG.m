
clc
clear
data=xlsread('exp.xlsx');
c=[];
alpha=0.5;
boxsize=1.5;
kk=1;
weighted=1;
cndm = condition_ndm(data,alpha,boxsize,kk);
