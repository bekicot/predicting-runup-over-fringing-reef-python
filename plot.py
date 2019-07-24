#%%
import pandas as pd
import pyperclip as clip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#%%
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [24, 40]

MIN = 0
MAX = 18000
LENGTH = MAX - MIN
TEST_NUMBER = '18'
FREQUENCY = 20

pd_data = pd.read_csv('data/UMtests_1-53/test-' + TEST_NUMBER + '.dat', header=7, sep='\t')
data_file = pd_data.get_values()[MIN:MAX]

x = np.arange(MIN, (MAX)/20, 1/20)
plt.subplots_adjust(hspace=0.5)
for min in range(1, 11):    
    plt.subplot(11, 1, min + 1)
    plt.title("Sensor " + str(min))
    plt.ylabel('Wave Height (cm)')
    plt.xlabel('Waktu (detik)')
    plt.ylim(-10, 12)
#     plt.xlim(600, 800)
    plt.plot(x, data_file[:, min-1:min],'g-', markersize=0, linewidth=0.5)
plt.savefig("../plots/sensor-1-9-test18.pdf")

# plt.subplot(10, 1, 2)
# plt.ylabel('Wave Height (cm)')
# plt.xlabel('Data sequence')
# plt.ylim(-3, 10)
# plt.plot(x, data_file[:, 9:10],'g-', markersize=2, linewidth=1)
# p1.plot(data_file,'bo--', markersize=3, linewidth=1)

# clip.copy(pd_data.loc[:, ['Wave1;', 'Wave2;', 'Wave3;', 'Wave9;']].describe(include="all").to_latex())

#%% [markdown]
# 
#%% [markdown]
# 
#%% [markdown]
# plt.axvline(x=[2, 32, 45, 85], color='r', linestyle='-')
# plt.axvline(x=y[1][0], color='r', linestyle='-')

#%%
plt.show()


