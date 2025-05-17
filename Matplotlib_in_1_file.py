# Matplotlib : is a low level graph plotting library in python that serves as a visualization utility.
# Install with cmd : pip install matplotlib

# Bar chart, Histogram chart, Line chart, Pie chart, Scatter plot, Box Plot are mostly used for analysis 

import matplotlib.pyplot as plt 

 # Bar chart , Selfmade data : 
'''
x = ["py2020","py2021","py2022","py2023","py2024"] # x axis data
y = ["30%","35%","50%","60%","80%"] # y axis data

Design1 = {'family':'serif','color':'blue','size':20} # Variable
Design2 = {'family':'serif','color':'darkred','size':15}
barclr = ['r','b','y','g','purple']

plt.bar(x,y, color = barclr, width = 0.5)

plt.xlabel("Python popularity in year", fontdict = Design2) # fontdict to set font properties for the title and labels.
plt.ylabel(" Percentage ", fontdict = Design2)
plt.title("Python popularity",fontdict = Design1)
plt.grid(color = 'blue', linestyle = '--', linewidth = 0.5) # Add gridlines

plt.show() # Display the chart

'''

# External data :
'''
import pandas as pd 

data = pd.read_csv("Food.csv")
grouped = data.groupby("Food")["Price"].sum( )
plt.bar(grouped.index,grouped.values) # pandas method
plt.show()

'''



 # Line chart , selfmade data :

'''
x = ["day1","day2","day3","day4","day5"]
y = [300,420,250,230,400]
y2 = [500,450,300,250,320]

plt.plot(x,y,marker = "o" , ms = 10, color = "green",label = "week1" )
plt.plot(x,y2,marker = "*" , ms = 10,  color = "red",label = "week2",alpha = 0.5 )
plt.legend() # Display indicator
plt.show()

'''


# External data : 

'''
import pandas as pd 

data = pd.read_csv("Food.csv")
grouped = data.groupby("Food")["Price"].sum()
plt.plot(grouped.index,grouped.values)
plt.show()

'''

 # Scatter chart , selfmade data :

'''
import numpy as np 

x = [5,7,8,7,2,17,2,9,4,11]
y = [99,86,87,88,111,86,103,87,94,78]

color = [0, 10, 20, 30, 40, 45, 50, 55, 60, 70] # For adding dot colors
sizes = [20,50,100,200,500,1000,60,90,10,300] # For adding dot sizes

plt.scatter(x,y,marker = "*",cmap = "viridis",c = color, s = sizes) # cmap = The Matplotlib module has a number of available colormaps. viridis = which is one of the built-in colormaps available in Matplotlib

plt.colorbar()
plt.show()

'''



 # External data :

'''
import numpy as np
import pandas as pd

data = pd.read_excel("ESD.xlsx")
plt.scatter(data["Age"],data["EEID"])
plt.show()

'''

 # Pie chart , selfmade data :

'''
brands = ["Oneplus","Apple","Samsung","Nokia","Redmi","Xio"]
popularity = [20,20,40,10,10,20]
c = ["blue","purple","cyan","skyblue","cadetblue","pink"]
ex = [0,0,0.1,0,0,0]

plt.pie(popularity, labels = brands, colors = c, explode = ex, autopct = "%.f")
# colors = set the color of each wedge, explode = you can separate any wedge, autopct = display values in wedge
plt.show()

'''


 # External data :

'''
import pandas as pd

data = pd.read_csv("Food.csv")
grouped = data.groupby("Food")["Popularity"].sum()
plt.pie(grouped.values, labels = grouped.index, autopct = "%.f")
plt.show()

'''

 # Box chart, selfmade data :

'''
a = [1,3,4,7,12,2,8,9,24]
b = [2,3,4,6,3,5,7,3,6]
c = [a,b]

plt.boxplot(c)
plt.show()

'''

 # External data :

'''
import pandas as pd

data = pd.read_excel("ESD.xlsx")
plt.boxplot(data["Annual Salary"])
plt.show()

'''


 # Histogram , selfmade data :

'''
import numpy as np

x = np.random.normal(170, 10, 250)

plt.hist(x)
plt.show()


You can read from the histogram that there are approximately:

2 people from 140 to 145cm
5 people from 145 to 150cm
& more .....

'''



 # External data :

'''
import pandas as pd

data = pd.read_excel("ESD.xlsx")
plt.hist(data["Age"], bins = 15)
plt.show()

'''

 # Violin chart , selfmade data :

'''
a = [20,30,40,50,0,30,40,40,30,70]

plt.violinplot(a, showmedians = True)
plt.show()

'''
 
 # External data :

'''
import pandas as pd

data = pd.read_excel("ESD.xlsx")
plt.violinplot(data["Annual Salary"], showmedians = True)
plt.show()

'''

 # Stem chart , selfmade data :

'''
x = [10,40,50,40,20,40,20,40,60,50,60]
plt.stem(x, linefmt = "--", markerfmt = "D") # For styling 
plt.show()

'''


 # External data :

'''
import pandas as pd

data = pd.read_excel("ESD.xlsx")
plt.stem(data["Age"].head(10))
plt.show()

'''

 # Stack chart, selfmade data :

'''
days = [1,2,3,4,5,6,7]

NOP1 = [5,10,30,20,35,60,80]
NOP2 = [50,60,30,75,80,90,120]
NOP3 = [8,30,50,60,70,90,100]

plt.stackplot(days, NOP1, NOP2, NOP3, colors = ["red","blue","green"], labels = ["week1","week2","week3"])
plt.xlabel("Days")
plt.ylabel("Values")
plt.legend()
plt.show()

'''

 # External data :

'''
import pandas as pd

data = pd.read_excel("Food nutrition.xlsx")
grouped = data.groupby("Category")[["Calories","Protein","Fat"]].agg("mean")


plt.stackplot(data["Category"].unique(), grouped["Calories"], grouped["Protein"], grouped["Fat"], labels = ["Calories","Protein","Fat"])
plt.legend()
plt.show()

'''

 # Step chart, selfmade dat :

'''
x = ["day1","day2","day3","day4","day5"]
y = [30,40,25,30,40]
plt.step(x,y, where = "post", marker = "o")
plt.show()

'''

 # External data : 

'''
import pandas as pd

data = pd.read_excel("ESD.xlsx")
grouped = data.groupby("Department").agg({"Annual Salary" : "mean"})
plt.step(grouped.index, grouped["Annual Salary"], where = "post", marker = "o")
plt.show()

'''

 # 3D plot :
''' 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create data
np.random.seed(42)  # For reproducibility
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)

# Create a figure and a 3D Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z, c='r', marker='o')

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show plot
plt.show()

'''


