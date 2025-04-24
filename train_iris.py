import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from id3 import ID3Classifier, print_tree_dot

# read the already discretized dataset 
df = pd.read_csv("iris_discretized.csv")

# separate features and labels 
x = df.drop("class", axis=1)
y = df["class"]

# divide into training (80%) and testing (20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# join attributes and target in only one dataframe 
df_train = x_train.copy()
df_train['target'] = y_train

# train the tree
tree = ID3Classifier()
tree.fit(df_train, 'target')

# generate and save the tree image 
print_tree_dot(tree, "iris_tree.dot")
os.system("dot -Tpng iris_tree.dot -o iris_tree.png")
print("Árvore guardada como 'iris_tree.png'")

# evaluate on the test set
correct = 0
total = len(x_test)
predictions = []

for i in range(total):
    example = x_test.iloc[i]
    predicted = tree.predict(example)
    predictions.append(predicted)
    actual = y_test.iloc[i]
    if predicted == actual:
        correct += 1

accuracy = correct / total
print(f"Precisão no conjunto de teste: {accuracy:.2%}")

# confusion matrix
print("Matriz de Confusão:")
print(confusion_matrix(y_test, predictions))

# classification report
print("\nRelatório de Classificação:")
print(classification_report(y_test, predictions))

# Show image
img = Image.open("iris_tree.png")
img.show()

input("Pressione Enter para fechar...")
