# ASOS purchase prediction task - Mitch Watts

## Problem statement
Come up with a solution that, given past data, predicts whether a customer will purchase a product after viewing a product one or multiple times.

## Code

### layout
The structure is:
|
prediction_code
    |
    evaluation
        |
        evaluate.py # functions for evaluating predictions and storing those 
                    # predicitons
    misc
        |
        caching.py # functions for caching files of all kinds
        constants.py # constants used elsewhere, as well as custom exceptions
    models
        |
        common_funcs.py # some functions useful to all model files
        baselines
            |
            normalised_popularity_baseline.py # call this to load (or compute and 
                                              # cache) predictions from normalised 
                                              # popularity baseline.
            random_baseline.py # call this to load (or compute and cache) 
                               # predictions from random baseline.
        collaborative_filtering
            |
            memory_based
                |
                knn.py # call this to load (or compute and cache) predictions from 
                       # knn model.
            model_based
                |
                coclustering.py # call this to load (or compute and cache) 
                                # predictions from coclustering model.
                nmf.py # call this to load (or compute and cache) predictions 
                       # from nmf model.
                slopeone.py # call this to load (or compute and cache) 
                            # predictions from slopeone model.
                svd.py # call this to load (or compute and cache) predictions 
                       # from svd model.
                nn
                    |
                    nn_misc.py # general functions for nn (lr plotter)
                    nn_models.py # model classes for NNs
                    nn.py # call this to load (or compute and cache) predictions 
                          # from collaborative filtering nn model.
                    nn_plots
                        | # plots from training are saved here
        content_based_filtering
            |
            customer_vector_similarity.py # call this to load (or compute and 
                                          # cache) predictions from customer 
                                          # vector similarity model.
            product_vector_similarity.py # call this to load (or compute and 
                                         # cache) predictions from product vector 
                                         # similarity model.
            nn
                |
                nn_misc.py # general functions for nn (lr plotter)
                nn_models.py # model classes for NNs
                nn.py # call this to load (or compute and cache) predictions from 
                      # content-based filtering nn model.
                nn_plots
                    | # plots from training are saved here
        hybrid
            |
            nn_misc.py # general functions for nn (lr plotter)
            nn_models.py # model classes for NNs
            nn.py # call this to load (or compute and cache) predictions from 
                  # hybrid nn model.
            nn_with_views.py # call this to load (or compute and cache) 
                             # predictions from hybrid nn model that incorporates 
                             # views data.
            nn_views_coclustering.py # call this to load (or compute and cache) 
                                     # predictions from hybrid nn model that 
                                     # incorporates views and uses coclustering 
                                     # instead of collaborative filtering arm.
            nn_plots
                | # plots from training are saved here
    plotting
        |
        compare_models.py # call this from cmd line to plot comparison of all 
                          # models
        plot_data.py # call from cmd line to make plots exploring raw data
        plot_evaluations.py # call from cmd line to make plots comparing all 
                            # models
    processing
        |
        data_loading.py # functions for loading raw data
        processors.py # functions for transforming raw data
        split_data.py # functions for splitting data into train val test
    recmetrics 
        | # slimmed down version of github package recmetrics - generally 
        | # wasn't very useful so just kept what I needed
        plots.py # functions for plotting class separation and prec-recall curves
|
data
    |
    model_files
        | # files to speed up training are cached here
    plots
        | # files for model evaluation and data exploration are saved here
    predictions
        | # predictions for each model are stored here
    raw_data
        | # all the data you sent in original formats also some gzips for train/
        | # val/test split
    scores
        | # dictionaries containing metrics for each model are saved here 
        | # individually, as well as the aggregate scores for all models (these 
        | # are saved with .json extensions for val and test)
    split_data
        | # the split datasets are cached here
|
requirements.txt # build venv using pip. i.e. python3 -m venv venv. Then 
                 # activate, then pip install -r requirements.txt

### notes

I wanted to code to run from anywhere, so you can see the little 'sys.path.append(PDIR)' in each file, this is just so there are no issues with relative imports.

### papers
