import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('DHT22_data.csv')
xx = df['Temperature'].values

plt.plot(xx)
plt.show()