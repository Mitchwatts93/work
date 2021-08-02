import os, sys

################################################################################

CDIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(CDIR)

sys.path.append(PDIR) # I couldn't be bothered with making it a package, 
# so I'm doing this to make sure imports work when run from anywhere

from processing import customers, labels_training, products, purchases, views

################################################################################
def load_all_dfs():
    products_df = products.get_products_df()
    customers_df = customers.get_customers_df()
    labels_training_df = labels_training.get_labels_training_df()
    purchases_df = purchases.get_purchases_df()
    views_df = views.get_views_df()

    all_dfs = {
        "customers":customers_df,
        "labels_training":labels_training_df,
        "products":products_df,
        "purchases":purchases_df,
        "views":views_df,
    }
    return all_dfs

################################################################################

def main():
    all_dfs = load_all_dfs()
    breakpoint()
    pass

if __name__ == "__main__":
    main()