import pandas as pandas
import math 
from collections import Counter
from PIL import Image  



class DecisionTreeNode:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute  # atributo usado neste nó
        self.label = label  # classe (se for folha)
        self.children = {}  # dicionário: valor do atributo -> subárvore

    def is_leaf(self):
        return self.label is not None

    def classify(self, example):
        if self.is_leaf():
            return self.label
        attr_value = example[self.attribute]
        if attr_value in self.children:
            return self.children[attr_value].classify(example)
        else:
            return None  # valor desconhecido

class ID3Classifier:
    def __init__(self):
        self.root = None

    def entropy(self, labels):
        total = len(labels)
        counts = Counter(labels)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    def info_gain(self, df, attribute, target_attribute):
        total_entropy = self.entropy(df[target_attribute])
        values = df[attribute].unique()

        weighted_entropy = 0
        for v in values:
            subset = df[df[attribute] == v]
            weight = len(subset) / len(df)
            weighted_entropy += weight * self.entropy(subset[target_attribute])

        return total_entropy - weighted_entropy

    def majority_class(self, labels):
        return Counter(labels).most_common(1)[0][0]

    def build_tree(self, df, target_attribute, attributes):
        labels = df[target_attribute]
        if len(set(labels)) == 1:
            return DecisionTreeNode(label=labels.iloc[0])
        if not attributes:
            return DecisionTreeNode(label=self.majority_class(labels))

        # choose attribute with highes information gain
        gains = {attr: self.info_gain(df, attr, target_attribute) for attr in attributes}
        best_attr = max(gains, key=gains.get)

        node = DecisionTreeNode(attribute=best_attr)

        for value in df[best_attr].unique():
            subset = df[df[best_attr] == value]
            if subset.empty:
                node.children[value] = DecisionTreeNode(label=self.majority_class(labels))
            else:
                new_attrs = [a for a in attributes if a != best_attr]
                node.children[value] = self.build_tree(subset, target_attribute, new_attrs)

        return node

    def fit(self, df, target_attribute):
        attributes = [col for col in df.columns if col != target_attribute]
        self.root = self.build_tree(df, target_attribute, attributes)

    def predict(self, example):
        return self.root.classify(example)

       
import graphviz

def print_tree_dot(tree, filename="tree.dot"):
    with open(filename, "w") as f:
        f.write("digraph Tree {\n")
        _print_tree_dot_helper(tree.root, f)  # <- usa tree.root
        f.write("}")
    print(f"Árvore guardada como '{filename}'")



def _print_tree_dot_helper(node, f, node_id = 0, counter=[1]):
    my_id = node_id 
    label = node.label if node.label is not None else "?"
    f.write(f'node{my_id} [label="{label}"];\n')

    for attr_val, child in node.children.items():
        child_id = counter[0]
        counter[0] += 1
        f.write(f'node{my_id} -> node{child_id} [label ="{attr_val}"];\n')
        _print_tree_dot_helper(child, f, child_id, counter)
