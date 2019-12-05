# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.17
 
# set height of bar
bars1 = [7,	18,	19,	4,	35,	4,	11,	7]
bars2 = [0,	11,	21,	5,	13,	0,	2,	6,]
bars3 = [1, 7,	11,	1,	7,	2,	3,	5]
bars4 = [3,	5,	6,	2,	12,	2,	7,	5]
 


# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
 
# Make the plot
plt.bar(r1, bars1, color='#000000', width=barWidth, edgecolor='black', label='Topic 1')
plt.bar(r2, bars2, color='#ffffff', width=barWidth, edgecolor='black', label='Topic 2')
plt.bar(r3, bars3, color='#c1c1c1', width=barWidth, edgecolor='black', label='Topic 3')
plt.bar(r4, bars4, color='#777777', width=barWidth, edgecolor='black', label='Topic 4')

        
#plt.figure(figsize=(20,5))

# Add xticks on the middle of the group bars
plt.xlabel('Coding', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['ACP', 'CMEASURE', 'CTYPE', 'DEPTH', 'METH', 'SQUAL', 'SRLP', 'WSAP'])
 
# Create legend & Show graphic
plt.legend()
plt.xticks(rotation=45)
plt.title('Surface Web - Topics distribution',fontsize=20)
plt.show()



#######################
"""VERSIONE 2 """

"""################## SURFACE WEB ###############"""
######################


# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

barWidth = 0.17
 
# Data



raw_data = {'topic1': [7,18,16,4,23,1,10,7], 'topic2': [1,11,20,5,19,6,9,6],'topic3': [2,4,13,2,15,0,3,6],'topic4': [1,8,8,1,10,1,1,4]}
df = pd.DataFrame(raw_data)

# Set position of bar on X axis
r1 = np.arange(len(df))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# From raw value to percentage
totals = [i+j+k+l for i,j,k,l in zip(df['topic1'], df['topic2'], df['topic3'], df['topic4'])]
#print(totals)
totale = sum(totals[0:len(totals)])    
#print(totale)

topic1 = [i / totale * 100 for i,j in zip(df['topic1'], totals)]
topic2 = [i / totale * 100 for i,j in zip(df['topic2'], totals)]
topic3 = [i / totale * 100 for i,j in zip(df['topic3'], totals)]
topic4 = [i / totale * 100 for i,j in zip(df['topic4'], totals)]
print(topic1, topic2, topic3, topic4)


# Make the plot
plt.bar(r1, topic1, color='#000000', width=barWidth, edgecolor='black', label='Topic 1')
plt.bar(r2, topic2, color='#ffffff', width=barWidth, edgecolor='black', label='Topic 2')
plt.bar(r3, topic3, color='#c1c1c1', width=barWidth, edgecolor='black', label='Topic 3')
plt.bar(r4, topic4, color='#777777', width=barWidth, edgecolor='black', label='Topic 4')

# Add xticks on the middle of the group bars
plt.xlabel('Thematic Coding', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(df))], ['ACP', 'CMEASURE', 'CTYPE', 'DEPTH', 'METH', 'SQUAL', 'SRLP', 'WSAP'])

plt.gca().set_yticklabels(['{:.0f}%'.format(x*1) for x in plt.gca().get_yticks()]) 



# Create legend & Show graphic
plt.legend()
plt.xticks(rotation=45)
plt.title('Surface Web - Topics distribution',fontsize=20)
plt.show()


###############################################################################
###############################################################################
###############################################################################
###############################################################################
"""################## VERSIONE 2 ###############"""

"""################## DEEP/DARK WEB ###############"""
######################

# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

barWidth = 0.17
 
# Data



raw_data = {'topic1': [0,12,10,4,18,0,2,0], 
            'topic2': [3,11,11,3,8,0,1,1],
            'topic3': [0,4,7,1,4,0,3,0],
            'topic4': [2,8,15,2,3,0,0,0]}

df = pd.DataFrame(raw_data)

# Set position of bar on X axis
r1 = np.arange(len(df))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# From raw value to percentage
totals = [i+j+k+l for i,j,k,l in zip(df['topic1'], df['topic2'], df['topic3'], df['topic4'])]
#print(totals)
totale = sum(totals[0:len(totals)])    
print(totale)

topic1 = [i / totale * 100 for i,j in zip(df['topic1'], totals)]
topic2 = [i / totale * 100 for i,j in zip(df['topic2'], totals)]
topic3 = [i / totale * 100 for i,j in zip(df['topic3'], totals)]
topic4 = [i / totale * 100 for i,j in zip(df['topic4'], totals)]
print(topic1, topic2, topic3, topic4)


# Make the plot
plt.bar(r1, topic1, color='#000000', width=barWidth, edgecolor='black', label='Topic 1')
plt.bar(r2, topic2, color='#ffffff', width=barWidth, edgecolor='black', label='Topic 2')
plt.bar(r3, topic3, color='#c1c1c1', width=barWidth, edgecolor='black', label='Topic 3')
plt.bar(r4, topic4, color='#777777', width=barWidth, edgecolor='black', label='Topic 4')

# Add xticks on the middle of the group bars
plt.xlabel('Thematic Coding', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(df))], ['ACP', 'CMEASURE', 'CTYPE', 'DEPTH', 'METH', 'SQUAL', 'SRLP', 'WSAP'])

plt.gca().set_yticklabels(['{:.0f}%'.format(x*1) for x in plt.gca().get_yticks()]) 



# Create legend & Show graphic
plt.legend()
plt.xticks(rotation=45)
plt.title('Deep/Dark Web - Topics distribution',fontsize=20)
plt.show()




###############################################################################
###############################################################################
###############################################################################
###############################################################################
"""################## VERSIONE 2 ###############"""

"""################## DEEP/DARK WEB 5 TOPICS ###############"""
######################

# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

barWidth = 0.17
 
# Data



raw_data = {'topic1': [2,17,23,8,19,0,3,0], 'topic2': [1,11,9,2,10,0,1,0],'topic3': [0,0,1,0,1,0,2,0],'topic4': [1,1,5,0,0,0,0,0],'topic5': [1,6,5,0,3,0,0,1]}
df = pd.DataFrame(raw_data)

# Set position of bar on X axis
r1 = np.arange(len(df))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

# From raw value to percentage
totals = [i+j+k+l+m for i,j,k,l,m in zip(df['topic1'], df['topic2'], df['topic3'], df['topic4'], df['topic5'])]
#print(totals)
totale = sum(totals[0:len(totals)])    
print(totale)

topic1 = [i / totale * 100 for i,j in zip(df['topic1'], totals)]
topic2 = [i / totale * 100 for i,j in zip(df['topic2'], totals)]
topic3 = [i / totale * 100 for i,j in zip(df['topic3'], totals)]
topic4 = [i / totale * 100 for i,j in zip(df['topic4'], totals)]
topic5 = [i / totale * 100 for i,j in zip(df['topic5'], totals)]

#print(topic1, topic2, topic3, topic4)


# Make the plot
plt.bar(r1, topic1, color='#000000', width=barWidth, edgecolor='black', label='Topic 1')
plt.bar(r2, topic2, color='#ffffff', width=barWidth, edgecolor='black', label='Topic 2')
plt.bar(r3, topic3, color='#c1c1c1', width=barWidth, edgecolor='black', label='Topic 3')
plt.bar(r4, topic4, color='#444444', width=barWidth, edgecolor='black', label='Topic 4')
plt.bar(r5, topic5, color='#777777', width=barWidth, edgecolor='black', label='Topic 5')


# Add xticks on the middle of the group bars
plt.xlabel('Coding', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(df))], ['ACP', 'CMEASURE', 'CTYPE', 'DEPTH', 'METH', 'SQUAL', 'SRLP', 'WSAP'])

plt.gca().set_yticklabels(['{:.0f}%'.format(x*1) for x in plt.gca().get_yticks()]) 



# Create legend & Show graphic
plt.legend()
plt.xticks(rotation=45)
plt.title('Deep/Dark Web - Topics distribution',fontsize=20)
plt.show()


















        
