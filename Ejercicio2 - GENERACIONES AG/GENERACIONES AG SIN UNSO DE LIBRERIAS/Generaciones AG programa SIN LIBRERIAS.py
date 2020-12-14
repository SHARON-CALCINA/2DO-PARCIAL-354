# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:56:43 2020

@author: SHARON
"""

# EJERCICIO 3 - 2DO PARCIAL

#.Dado la función y= x3+x2+x, en excel realice al menos tres generaciones del
#funcionamiento del algoritmo genético,
lista= [6,7,10,26,16,12,15,24,20,19,5,13,4,28]
def evalFun(x):
    return (pow(x, 3)+pow(x,2)+x)
def decBin(x):
    x=x[::-1]
    sum=0
    for i in range (0, len(x)):
        val= x[i]
        if val =='1':
            sum+= pow(2,i)
    return sum
print(decBin('011000'))

import pandas as pd
import numpy as np
def muestraDatos(m):
    res= pd.DataFrame(m)
    res=res.T
    res.columns=['x','x_ordenado','f(x)=x3+x2+x','binario_x','complemento',
                'cruce','mutacion','valor']
    print(res)

def algGenetic(generacion, lista):
    
    for i in range (1, generacion+1):
        resultado=[]
        
        l= [str(int) for int in lista]
        resultado.append(l)
        print('------------------- GENERACION ', i,' ----------------------')
        print('*********   x   *********')
        print(lista)
        fun=[]
        binario=[]
        lista= sorted(lista, reverse=True)
        for i in range(0, len(lista)):
            fun.append(evalFun(lista[i]))
            binario.append(bin(lista[i]))
        
        print('*********   x ordenado   *********')
        print(lista)
        print('*********   f(x)= x3+x2+x   *********')
        print(fun)
        print('*********   binario de x   *********')
        lo= [str(int) for int in lista]
        resultado.append(lo)
        f= [str(int) for int in fun]
        resultado.append(f)
        tamPrim= lista[0]
        binario2=[]
        tamPrim=tamPrim.bit_length()+1
        
        for i in range (0,len(binario)):
            binario2.append(''+(binario[i]))
        for i in range (0,len(binario2)):
            num = binario2[i].split('b')
            binario2[i] = num[1]
        
        b= [str(int) for int in binario2]
        print(binario2)
        resultado.append(b)
        for i in range (0,len(binario2)):
            while(len(binario2[i])<tamPrim):
                binario2[i]= '0'+binario2[i]
        print('*********   complemento   *********')
        print(binario2)
        c= [str(int) for int in binario2]
        resultado.append(c)
        for i in range(0,len(binario2),2):
            val1=binario2[i][3:]
            val2=binario2[i+1][3:]
            binario2[i]= ''+binario2[i][0:3]+val2
            binario2[i+1]= binario2[i+1][0:3]+val1
        print('*********   cruce   *********')
        print(binario2) 
        cr= [str(int) for int in binario2]
        resultado.append(cr)
        for i in range(0,len(binario2)):
            val=binario2[i][3]
            if val =='1':
                val='0'
            else: 
                val='1'
            binario2[i]= binario2[i][0:3]+val+binario2[i][4:6]
        print('*********   mutacion   *********')
        print(binario2)
        m= [str(int) for int in binario2] 
        resultado.append(m)
        for i in range(0,len(binario2)):
           binario2[i]=decBin(binario2[i])
        print('*********   valor   *********')
        print(binario2)
        v= [str(int) for int in binario2]
        resultado.append(v)
        lista=binario2
        muestraDatos(resultado)
    
algGenetic(6, lista)


    
        
