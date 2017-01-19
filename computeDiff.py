import numpy as np

fileNameA = "TClog/cpu_1/merged_conv_2_data"
#fileNameA = "TClog/cpu_1/postConv_blobVec_1_data"
fileNameB = "TClog/gpu_2/postConv_data_gpu_transition1"
A_offset = 125
B_offset = 125
rangeLen = 50

listA =  open(fileNameA,'r').readlines()[0].split(",")[:-1]
listB =  open(fileNameB,'r').readlines()[0].split(',')[:-1]
floatAL = map(lambda x:float(x),listA)
floatBL = map(lambda x:float(x),listB)


print len(floatAL)
print len(floatBL)
print floatAL[A_offset:A_offset+rangeLen]
print floatBL[B_offset:B_offset+rangeLen]
"""
for i in range(rangeLen):
    numA,numB = floatAL[A_offset+i],floatBL[B_offset+i]
    if abs(numA-numB)>0.01:
        print `numA`+":"+`numB`+":"+`numA-numB`
        print i
"""
globalMaxDiff = 0 
for i in range(rangeLen):
    aIdx,bIdx = A_offset+i,B_offset+i 
    globalMaxDiff = max(globalMaxDiff,abs(floatAL[aIdx]-floatBL[bIdx]))

print "global Max Diff is:"+`globalMaxDiff`


