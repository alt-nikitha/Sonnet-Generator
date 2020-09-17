import re
import sys
#rawtext=open('dataset.txt').read()
#print(rawtext)
f=open('preprocess.txt','w')
for line in open(sys.argv[1]):
    
    line1beg=re.sub(r"<eos> ", "\n", line)
    line2=re.sub(r"<eos>","\n",line1beg)
    # line1end=re.sub(r"<eos>","<EndOfSonnet>",line[-7:])
    # line2="<StartOfSonnet>"+line[:-7]+line1end
    f.write(line2)
    
for line in open(sys.argv[2]):
    line1beg=re.sub(r"<eos> ", "\n", line)
    line2=re.sub(r"<eos>","\n",line1beg)
    # line1end=re.sub(r"<eos>","<EndOfSonnet>",line[-7:])
    # line2="<StartOfSonnet>"+line[:-7]+line1end
    f.write(line2)
