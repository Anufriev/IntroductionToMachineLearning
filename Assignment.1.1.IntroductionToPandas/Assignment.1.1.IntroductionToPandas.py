import pandas

def createSubmissionFile(key, submission):
  fo = open('Submissions/' + key + '.txt', 'w')
  fo.write(str(submission))
  fo.close()

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

print '1. How many men and women rode on the ship?'
result = data['Sex'].value_counts()
print result
men = result['male']
women = result['female']
submission = str(men) + ' ' + str(women)
print submission
print
createSubmissionFile('1. How many men and women rode on the ship', submission)


print '2. What part of the passengers managed to survive?'
result = data['Survived'].value_counts(sort=False)
print result
died = result[0]
survived = result[1]
percentage = round(float(survived * 100) / (died + survived), 2)
print percentage
print
createSubmissionFile('2. What part of the passengers managed to survive', percentage)

print '3. What percentage of first class passengers were among the passengers?'
result = data['Pclass'].value_counts(sort=False)
print result
percentage = round(float(result[1] * 100) / (result[1] + result[2] + result[3]), 2)
print percentage
print
createSubmissionFile('3. What percentage of first class passengers were among the passengers', percentage)

print '4. What age were the passengers? Calculate the average and median age of the passengers.'
averageAge = data['Age'].mean()
medianAge = data['Age'].median()
submission = str(round(averageAge, 2)) + ' ' + str(medianAge)
print submission
print
createSubmissionFile('4. What age were the passengers', submission)

print '5. Pearson correlation between the SibSp sign and the Parch sign.'
result = round(data['SibSp'].corr(data['Parch']), 2)
print result
createSubmissionFile('5. Pearson correlation between the SibSp sign and the Parch sign', result)

print '6. What is the most popular female name on the ship?'

def extractFirstName(name):
  return name.partition('.')[2]
 
firstNames = data[data['Sex'] == 'female']['Name'].apply(extractFirstName)
result = firstNames.value_counts()
print result
createSubmissionFile('6. Most popular female name on the ship', result)