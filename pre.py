import re
import sys


#open a file to store the preprocessed text
f=open('preprocess.txt','w')

for line in open(sys.argv[1]): #for first dataset
    #each line has 13 "<eos> " sequences and 1 "<eos>" (without space)
    #replace this with a newline character
    line1beg=re.sub(r"<eos> ", "\n", line) 
    line2=re.sub(r"<eos>","\n",line1beg)
    #write each line to preprocessed file
    f.write(line2)
    
for line in open(sys.argv[2]): #for second dataset
    #same as previous
    line1beg=re.sub(r"<eos> ", "\n", line)
    line2=re.sub(r"<eos>","\n",line1beg)
    
    f.write(line2)
