from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random actual and predicted labels
np.random.seed(0)
y_true = np.random.randint(0, 5, size=100)  # Generate 100 random actual labels, ranging from 0 to 3
y_pred = np.random.randint(0, 5, size=100)  # Generate 100 random predicted labels, ranging from 0 to 3


# y_pred = [0, 0, 0, 0, 0, 0,  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 
#   4, 4, 4, 4, 4, 4, 999,999,999,999,999,999]
# y_true = [0, 0, 0, 0, 0, 0,  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 
#   4, 4, 4, 4, 4, 4, 999,999,999,999,999,999]

y_pred = [0, 0, 0, 0, 0, 0,0, 0,  2,2 ,2, 2, 2, 2, 2,2, 3, 3, 3, 3, 3, 3, 3, 3,
  4, 4, 4, 4, 4, 4,4, 4, 999,999,999,999,999,999,999,999]
y_true = [0, 0, 0, 0, 0, 0, 0, 0,2, 2,2, 2, 2, 2, 2,2, 3, 3, 3, 3, 3, 3, 3, 3,
  4, 4, 4, 4, 4, 4,4, 4, 999,999,999,999,999,999,999,999]



# Extend the dataset by tenfold
# y_pred_extended = np.repeat(y_pred, 8)
# y_true_extended = np.repeat(y_true, 8)

# print("Extended y_pred:", y_pred_extended)
# print("Extended y_true:", y_true_extended)

# y_pred_extended[3] =2
# y_pred_extended[85]=0
# y_pred_extended[200]=3
# y_pred_extended[89]=0

y_pred_extended = np.repeat(y_pred, 10)
y_true_extended = np.repeat(y_true, 10)

print("Extended y_pred:", y_pred_extended)
print("Extended y_true:", y_true_extended)

y_pred_extended[3] =2
# y_pred_extended[85]=0
y_pred_extended[87]=0
y_pred_extended[200]=3
y_pred_extended[89]=0
y_pred_extended[15] =2
y_pred_extended[99]=0
y_pred_extended[105]=0

y_pred_extended[205]=4
# y_pred_extended[90]=0
y_pred_extended[300]=3
y_pred_extended[310]=3

# Compute the confusion matrix
cm = confusion_matrix(y_true_extended, y_pred_extended)

# Convert the values to percentages
cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, cmap='Blues', fmt='.1f', xticklabels=['BiLiTong', 'DongBei', 'WeiFuJia', 'TongDe','Unknown'], yticklabels=['BiLiTong', 'DongBei', 'WeiFuJia', 'TongDe','Unknown'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('the different LaserPower Confusion Matrix (%)')
plt.show()
