import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

web_stats = {'Days': [1,2,3,4,5,6],
             'Visitors':[43,53,34,45,64,34],
             'Bounce_Rate':[65,72,62,64,54,66]}

df = pd.DataFrame(web_stats)

#print(df)
#print(df.head())
#print(df.tail())
#print(df.tail(2))

#df.set_index("Days", inplace=False)

#print(df['Visitors'])
#print(df.Bounce_Rate)
#Cannot use space in variable in df


print(df[['Visitors', 'Bounce_Rate']])

'''
Error cannot have multi-dimen array
lst = df[['Visitors', 'Bounce_Rate']].tolist()

Sol: Use numpy array
'''
#print(df.Visitors.tolist())
#Convert to list
print(np.array(df[['Visitors', 'Bounce_Rate']]))

df2 = pd.DataFrame(np.array(df[['Visitors', 'Bounce_Rate']]))

print(df2)

