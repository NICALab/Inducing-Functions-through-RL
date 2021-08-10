import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

flights = sns.load_dataset('flights')
sns.lineplot(data=flights, x='year', y='passengers')
sns.lineplot(data=flights, x='year', y='passengers')
plt.legend(['a', 'b'])
plt.title('asd', fontsize=17)
plt.xlabel("iteration", fontsize=15)
plt.xticks(fontsize=13)
plt.ylabel('asdasd', fontsize=15)
plt.yticks(fontsize=13)
plt.show()

df = pd.DataFrame({'A': 1.,
                   'B': 2.,
                   'C': np.zeros((100,))})
print(df)
print(df.dtypes)