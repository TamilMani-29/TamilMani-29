import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import model_selection


#importing dataset

df=pd.read_csv('student-por.csv',';')

#modifying dataset

df=df[['studytime','failures','absences','G1','G2','G3','internet','romantic','health']]
df=df.replace({'romantic':{'yes':1,'no':0}})
df=df.replace({'internet':{'yes':1,'no':0}})

#defining labels and attributes
x=np.array(df.drop(['G3'],1))
y=np.array(df['G3'])
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.1)


#using Linear regression algorithm
clf=LinearRegression()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)

#predicting the score through user input

res=[]
l=[]
text=['Enter the Study time of the student in hours each day','Enter the total number of failures of the student in the previous exams','Enter the number of days of the student\'s absence','Enter the mark obtained by the student in G1 Exam on a scale of 1-20','Enter the marks obtained by the student in G2 Exam on a scale of 1-20','Inernet Availability:\n\tEnter 0 for No\n\tEnter 1 for Yes','Does the student has any romantic relationship:\n\tEnter 1 for yes\n\tEnter 0 for No','Rate the health of the student on a scale of 1-5']
for i in range(8):
    print(text[i])
    user_input=int(input())
    l.append(user_input)
res.append(l)
marks=clf.predict(res)
marks=int(marks)
print('The predicted Score of the Student in the final Exam on a scale of 1 to 20 is: ',int(marks))

    

