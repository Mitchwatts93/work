# ASOS purchase prediction task - Mitch Watts

## Problem statement
Come up with a solution that, given past data, predicts whether a customer will purchase a product after viewing a product one or multiple times.

### Initial notes

- The statement doesn't specify some details. e.g. should this be a single model? It seems to hint that it should be, despite us having many customers. It would likely be intractable to have a model per customer, although in reality if using a NN we would probably have a customer embedding that could be useful. Because it would be required in production I will aim to make a single model, i.e. not a model fine tuned per customer, and try to encode the customer information in that model somehow.

- This should generalise to new customers, however there is no mention of the cold start problem. I will assume for now that we would have similar amounts of data for new customers, however thats unlikely in production.


## Code

### layout
The structure is:
|
code
    |
    processing
        |
        customers.py
        data_loading.py
        defaults.py
        labels_predict.py
        labels_training.py
        products.py
        purchases.py
        views.py
|
data
    |
    plots
    |
    raw_data
        |
        all the data you sent in original formats

I made a processing file for each piece of data, however the general transformations are also at the same level in the relevant files.

Plots will be referred to in this readme.

### notes

I wanted to code to run from anywhere, so you can see the little 'sys.path.append(PDIR)' in each file, this is just so there are no issues with relative imports.

## data exploration

### data formatting and initial impressions
the columns of the data depends on the data type. For some, these are bools, others are id numbers, some are year of birth, some are categorical (e.g. country, brand, producttype). We have ordinal floats e.g. prices also, we have ordinal ints e.g. viewOnly.

some of these are not useful features. e.g. if we want a model to generalise to new customers and new products, then using directly the customerId or the productId is not useful, but rather we can use them as identifiers. A simple way to do this for now would be to have a lookup table for productid, and that will be a vector of attributes about the product. This is a very sparse approach to the problem.

For most of the time features, I think some feature engineering would be useful. e.g. the current time of year might be useful, but really what we are interested in is time since the point of purchase. e.g. if the customer has been on the platform for 2 weeks and made 10 separate purchases, they are probably more likely to buy any given product than a customer that has been on the platform for 10 years and made 10 purchases.

product information is only useful relative to the customer. e.g. if the customer typically buys only a certain brand, the probability of purchasing any given product should be higher if it is the same brand. also goes for type of product.

the views_df and purchases_df will help us link customers and products.
the products_df will help with encoding product information.
the customers_df will help with encoding customer information.

The core of the problem is a recommendation problem, even though the labels_predict seems a bit different to this. This problem is a collaborative filtering problem.

As an intial note, a nice system would probably be end-to-end, using NNs. i.e. have a network for encoding products, a network for encoding customers, and then a network which takes the embedding of a product and a customer and calculates a probability of purchase. Then to train it we would simply input customer and product and use the true value bool as the label, so have a sigmoid at the output layer (only one class - buy probability, so no softmax). But how to encode new customers? the input could just be sparse for products - i.e. one hot encoding of the features of the product (brand, price, producttype, onsale, days_on_site??), similarly for customer. But then how to incorporate the customer history specifically into the input? It could be encoded in the system, but that isn't great... we could try encoding each interaction from views_df, and each purchase, but then how to input that - because the shape will vary? Alternatively, its simple to a very basic collaborative filtering using purchases_df only.

easy wins:
- if a customer has been looking at a product, then its probably a higher probability of buying
- if the customer has already purchased it, then there is a low probability of purchase

expected results:
- if the customer always buys similar brands, higher probability
- if the customer has previously bought a very similar product, lower probability??? not sure...
- if the customer never buys in that price range, lower probability
- if the product is on sale, probability should be higher than of a very similar product?


## relevant literature

### blogs 
https://www.kdnuggets.com/2017/08/recommendation-system-algorithms-overview.html

### papers
