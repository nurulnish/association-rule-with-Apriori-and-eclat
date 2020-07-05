# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import seaborn as sns
from apyori import apriori  
import csv 
from TransactionEncoder import TransactionEncoder

#Membaca data CSV----------------------------------
def readData(data_path):
 with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = list(reader)
 return data    
#----------------------------------------------------------
#Visualisasi Data Frequency dengan menggunakan seaborn dan pandas, import data csv di sini (ganti nama file di sini yah)
def visualization(data):
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df=df.applymap(lambda x: 1 if x==True else 0)
    myData=df.apply(pd.value_counts).T.sort_values(by=[1],ascending=False).head(20)
    Names=myData.index
    Freq=myData[1].values
    sns.barplot(x=Names,y=Freq)
path='data.csv'
data=readData(path)
#Memanggil algoritma apriori dari kelas apyori

# disini tempat ngubah min_sup sama min_conf
association_rules = apriori(data, min_support=0.001, min_confidence=0.001)  
#mengubah assosiations menjadi list
association_results = list(association_rules)  

#hasil di sini ya
print("We find "+str(len(association_results))+" assosiation rule.")  
#print(association_results)  
result=open("myresAPriory.txt", "w")
#visualisasi(data)

result.write("Rules\t Support \t Confidence \t Lift")
for item in association_results:
    pair = item[2][0][0]  
    items_1 = [x for x in pair]
    pair = item[2][0][1]  
    items_2 = [x for x in pair]
#    print("Rule: " + str(items_1)+ " -> " +str( items_2))
#    print("Support: " + str(item[1]))
#    print("Confidence: " + str(item[2][0][2]))
#    print("Lift: " + str(item[2][0][3]))
#    print("=====================================")
    
    result.write("\n"+ str(items_1)+ " -> " +str( items_2))
    result.write("\t" + str(item[1]))
    result.write("\t" + str(item[2][0][2]))
    result.write("\t" + str(item[2][0][3]))
#for item in association_results:
 #   result.write(str(item))
  #  result.write('----------------------------------------------'+'\n')
#    pair = item[0] 
#    items = [x for x in pair]
#    print("Rule: " + items[0] + " -> " + items[1])
#    print("Support: " + str(item[1]))
#    print("Confidence: " + str(item[2][0][2]))
#    print("Lift: " + str(item[2][0][3]))
#    print("=====================================")
result.close()
