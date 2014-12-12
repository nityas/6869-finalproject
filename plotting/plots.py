import matplotlib.pyplot as plt

#vary regularization
# plt.plot([0.01,0.05,5,10],[0.445627,0.446665,0.471062, 0.452894], 'r', [0.01,0.05,5,10],[0.915109,0.902648,0.850467, 0.775441], 'b')
# plt.xlabel('regularization parameter')
# plt.ylabel('accuracy')
# plt.title('Effect of Regularization Parameter on Accuracy')
# plt.show()

#vary feature map
plt.plot([5,7,10,15],[0.378666,0.419154,0.450298, 0.445627], 'r',[5,7,10,15], [0.827882,0.872793,0.907321,0.915109], 'b')
plt.xlabel('feature map dimension(number of features is square of this value)')
plt.ylabel('accuracy')
plt.title('Effect of Feature Map Size on Accuracy')
plt.show()

