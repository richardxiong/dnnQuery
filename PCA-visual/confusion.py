import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# array = np.loadtxt("./last_hidden_state-3300.txt", delimiter=',')
array = np.array([[33,2,0,0,0,0,0,0,0,1,3], 
        [3,31,0,0,0,0,0,0,0,0,0], 
        [0,4,41,0,0,0,0,0,0,0,1], 
        [0,1,0,30,0,6,0,0,0,0,1], 
        [0,0,0,0,38,10,0,0,0,0,0], 
        [0,0,0,3,1,39,0,0,0,0,4], 
        [0,2,2,0,4,1,31,0,0,0,2],
        [0,1,0,0,0,0,0,36,0,2,0], 
        [0,0,0,0,0,0,1,5,37,5,1], 
        [3,0,0,0,0,0,0,0,0,39,0], 
        [0,0,0,0,0,0,0,0,0,0,38]])

query = "ABCDEFGHIJK"
logic = "abcdefghijk"
# column: x-axis label; index: y-axis label
df_cm = pd.DataFrame(array, index = [i for i in query],
                  columns = [i for i in logic])

#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)  #for label size
sn.heatmap(df_cm, annot=True)
plt.show()
plt.savefig("./output.png")