#import caffe
import numpy as np
import matplotlib.pyplot as plt
import csv
import h5py
import os

path=os.path.dirname(os.path.abspath(__file__))     
i =0

train_array=np.ndarray(shape=(42000,1,28,28))
labels_array=np.ndarray(shape=(42000,1))
labels_array2=np.ndarray(shape=(28000,1))
test_array=np.ndarray(shape=(28000,1,28,28))

with open("../train.csv","r") as f:
    csv_r=csv.reader(f)
    csv_r.next()
    for row in csv_r:
        image1=row[1:]
        A=np.reshape(image1,(1,28,28)).astype(np.float32)
        A=A*(1/255.0)
        train_array[i][...]=A
        labels_array[i][...]=row[0]
        #plt.axis('off')
        #plt.imshow(A[0],cmap="Greys_r")
        #plt.savefig(str(i))
        #f_2.write('../examples/ishan/imageset/'+str(i)+'.png'+' '+row[0]+'\n')
        i+=1

    #f_2.close()
       
i=0
with open("../test.csv","r") as f:
    csv_r=csv.reader(f)
    csv_r.next()
    for row in csv_r:
        A=np.reshape(row,(1,28,28)).astype(np.float32)
        A=A*(1/255.0)
        test_array[i][...]=A
        labels_array2[i][...]=1
        i+=1


print train_array[0][0]
print labels_array
print test_array[0][0]
print labels_array2


with h5py.File("train.hdf5","w") as f_h5:
    f_h5['data']=train_array
    f_h5['label']=labels_array

with h5py.File("test.hdf5","w")as f_h52:
    f_h52['data']=test_array
    f_h52['label']=labels_array2
    
with open(path+'train.txt','w') as f:
    f.write(path+'train.hdf5'+'\n')
    
with open(path+'test.txt','w') as f:
    f.write(path+'test.hdf5'+'\n')

        
        
