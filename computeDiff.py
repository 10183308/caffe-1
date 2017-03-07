import numpy as np

fileNameA = "TClog/gpu_2/postBN_4G_data_transition1"
fileNameB = "TC_TrueFwdlog/cpu_1/postBN_BCVec_1_data"
A_offset = 0
B_offset = 0
rangeLen = 400

listA =  open(fileNameA,'r').readlines()[0].split(",")[:-1]
listB =  open(fileNameB,'r').readlines()[0].split(',')[:-1]
floatAL = map(lambda x:float(x),listA)
floatBL = map(lambda x:float(x),listB)


print len(floatAL)
print len(floatBL)

#print floatAL[A_offset:A_offset+rangeLen]
#print floatBL[B_offset:B_offset+rangeLen]

print "246 is:"
print `floatAL[246]`+","+`floatBL[246]`

for i in range(rangeLen):
    numA,numB = floatAL[A_offset+i],floatBL[B_offset+i]
    if abs(numA-numB)>0.034:
        print `numA`+":"+`numB`+":"+`numA-numB`
        print i

globalMaxDiff = 0 
for i in range(rangeLen):
    aIdx,bIdx = A_offset+i,B_offset+i 
    globalMaxDiff = max(globalMaxDiff,abs(floatAL[aIdx]-floatBL[bIdx]))

print "global Max Diff is:"+`globalMaxDiff`


