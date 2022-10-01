# Hung-yi Lee Machine Learning course HW4
Using Saliency map, filter explanation, lime to explain a food classification model. This dataset is collected from Kaggle 'Food-11 image dataset':

https://www.kaggle.com/datasets/trolukovich/food11-image-dataset 

It has in total 11 classes: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. To complete the classification task, we first build a simple CNN model, then train it and save its parameters. After that, we'll detect how machine can classify food using Saliency map, Filter visualization, and Lime methods.

## environment
torch == 1.10.1+cuda10.2

Lime == 0.0.1.37
