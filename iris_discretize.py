import pandas as pd 

# uploads dataset 
df = pd.read_csv("iris.csv")

# show first lines to confirm
print(df.head())

# function to discretize a column in 3 categories 
def discretize_column(df, column):

    # garantee that the column values are numerical 
    df[column] = pd.to_numeric(df[column], errors = 'coerce')

    # remove lines with non numeric values (converted to NaN)
    df = df.dropna(subset = [column])

    # apply discretization
    df[column] = pd.cut(df[column], bins = 3, labels = ["low", "medium", "high"])
    return df

# apply discretization to all numerical attributes
for col in df.select_dtypes(include ='number').columns:
    df = discretize_column(df, col)

print(df.head())

df.to_csv("iris_discretized.csv", index = False)