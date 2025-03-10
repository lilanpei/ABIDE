import matplotlib.pyplot as plt
import numpy as np

history_txt="H:\Google Drive\ABIDE\history\\new_20190329_AUC_history_MFC"
data = np.loadtxt(history_txt, delimiter=',', usecols = (2,3))
diff = data[:,1] - data[:,0]
#print(data[:,1])
#print(data[:,0])
print(len(diff))
diff_8 = list()
diff_16 = list()
diff_24 = list()
diff_32 = list()
diff_40 = list()
diff_48 = list()
for i in range(0,len(diff),6):
    diff_8.append(diff[i])
    diff_16.append(diff[i+1])
    diff_24.append(diff[i+2])
    diff_32.append(diff[i+3])
    diff_40.append(diff[i+4])
    diff_48.append(diff[i+5])
mean1 = np.mean(diff_8)
std1 = np.std(diff_8)
print("k = 8; AUC_DIFF = %0.3f ± %0.3f" % (mean1,std1))
mean2 = np.mean(diff_16)
std2 = np.std(diff_16)
print("k = 16; AUC_DIFF = %0.3f ± %0.3f" % (mean2,std2))
mean3 = np.mean(diff_24)
std3 = np.std(diff_24)
print("k = 24; AUC_DIFF = %0.3f ± %0.3f" % (mean3,std3))
mean4 = np.mean(diff_32)
std4 = np.std(diff_32)
print("k = 32; AUC_DIFF = %0.3f ± %0.3f" % (mean4,std4))
mean5 = np.mean(diff_40)
std5 = np.std(diff_40)
print("k = 40; AUC_DIFF = %0.3f ± %0.3f" % (mean5,std5))
mean6 = np.mean(diff_48)
std6 = np.std(diff_48)
print("k = 48; AUC_DIFF = %0.3f ± %0.3f" % (mean6,std6))
mean = np.asarray([mean1,mean2,mean3,mean4,mean5,mean6])
std = np.asarray([std1,std2,std3,std4,std5,std6])
k = np.asarray([8,16,24,32,40,48])
plt.errorbar(k, mean, std, linestyle='None', marker='o')
plt.title('Male/Female Classification')
plt.ylabel('(AUC_corrected_crossentropy - AUC_crossentropy)')
plt.xlabel('k')
plt.show()

