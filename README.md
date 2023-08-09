# PROJECT4

## Obtaining the Data

* We used the following dataset from Kaggle:

    * US College & University Admissions 2020-2021:  Admission Rates, SAT and ACT scores <br><br>

* Source:  https://www.kaggle.com/datasets/jfschultz/us-college-admisions-2021-rates-and-test-scores?resource=download

* We downloaded the data from Kaggle as a CSV file (Resources/df.csv).    

__About the Dataset__

    The dataset is compiled from The Integrated Postsecondary Education Data System (IPEDS) https://nces.ed.gov/ipeds/use-the-data

    IPEDS (Integrated Postsecondary Education Data System) is an annual data collection distributed by the Postsecondary Branch of the National Center for Education Statistics (NCES), a non-partisan center of the Institute of Education Sciences under the U.S. Department of Education. NCES is the primary federal entity for collecting and analyzing data related to education in the U.S. and other nations.

    Of the 6,289 institutions, listed with IPEDS, institutions which offer a four-year program conferring an undergraduate degree were included (n=2,556).

    Dataset was limited to institutions reporting numbers for both number of applicants and admissions, and enrolment size greater than 100 (n=1,386)

## Cleaning the Data

* First, we imported the CSV file into a jupyter notebook (**education1.ipynb**).
* We renamed some of the column headers to make them more human readable, and to remove any spaces that would create problems later on with SQL.
* Then, we changed the null values to zero (0) and changed the data type of the four SAT score seies from float to integer.
* Next we dropped "Other US Jurisdctn" and "US Service Schools" from the Region column. This was done to normalize the data, as these two categories applied different acceptance criteria than the majority of the population.
* We then saved the cleaned dataframe to a new CSV file (education_cleaned.csv).

## Reading the Data into SQL

* We used the existing database with the cleaned data to create a new database using SQLite (**schools_db**).
* We created a table (schools), then imported the data from the cleaned dataframe into the table.
* To double check that the import was successful, we printed the rows in the "schools" table.


    **Note:**  The Keras Tuner and Autohyper models (**ACT_Colab_Autohyper.ipynb**, **SAT_Colab_Autohyper.ipynb**, **ACT_KT_Colab.ipynb**, and **SAT_KT_Colab.ipynb**) used the original schools_db SQL database and schools table. During the model development stage, it was determined that some models functioned better with nulls removed entirely, instead of converted to zero (0). This was accomplished by creating a second version of the jupyter notebook (education2.ipynb). The new SQL database was named "school_data_db" and the new table was named "school_data". We used the school_data_db SQL database and school_data table for the K-means model (**K_means_SAT_clustering.ipynb**).

## Clustering the Schools Using K-Means Model (K_means_SAT_clustering.ipynb)

* Imported the data from the "school_data_db" SQL database and read in the "school_data" table
* Created a dataframe from the school_data table
* Dropped columns with string values and other columns deemed unnecessary
* Scaled the data using the StandardScaler function
* Created variables for K-values and inertia
* Fit the K-means model to the scaled data
* Ploted an eblow curve to determine the best K-value, which we determined was **3**
* We then initialized the K-means model to group the data into **3** clusters and fit the model to the scaled data
* Printed an array of the model predictions
* We then added the cluster array to the scaled dataframe as a new column
* Then, we created 3 scatter plots to show the following relationships:
    * Admissions Rate vs. SAT Math Score 75th Percentile
    * Admissions Rate vs. SAT Verbal Score 75th Percentile
    * SAT Math Score 75th Percentile vs. SAT Verbal Score 75th Percentile

## Creating Models Using Autohyper (ACT_Colab_Autohyper.ipynb and SAT_Colab_Autohyper.ipynb)

* Created a method that creates and compiles a new Sequential deep learning model with hyperparameter options.
* Allowed KerasTuner to select between **ReLU**, **Sigmoid**, and **tanh** activation functions for each hidden layer.
* Allowed KerasTuner to decide from 1 to 20 neurons in the first dense layer.
* Allow KerasTuner to decide from 1 to 4 hidden layers and 1 to 20 neurons in each Dense layer.
* Imported the KerasTuner library and create a **Hyperband** tuner instance. Use the following parameters in your tuner with max_opochs = to 50 and hyperband_iterations=2.
* Ran the KerasTuner search for best hyperparameters over 50 epochs.
* Retrieved the best model hyperparameters from the tuner search and printed the values.
* Printed the best model summary

## Creating Models Using Keras Tuner (ACT_KT_Colab.ipynb and SAT_KT_Colab.ipynb)

* Created a Keras Sequential model and added **3** dense hidden layer to create a deep learning model.
    * NOTE: Tried 4 dense hidden layers but that was too much.
* As with our activity models, the first Dense layer used the *input_dim* parameter.
* All of the hidden layers used the **ReLU** activation function.
    * NOTE: Didn’t try other activation functions since we were going to design the automatic hyperparameter models in another notebook.
* Compiled each model and trained the deep learning model on 100 epochs and validation_split at .15
* Evaluated the performance of the models by calculating the loss and predictive accuracy of the models on the test dataset.

## Creating Model Visualizations Using Seaborn Pairplot

* Use the seaborn library for plotting **pairplot** to take a look of the relationship between every single variable or feature in the dataframe
* This gives a view of the relationship between dependent (y) and independent variables (X)
* Created and plot the correlation heatmap / get the correlations of the data

## Building the Dense Neural Network

* Train linear regression models (the data the model has not seen before) and evaluate the model (getting the accuracy score)
* Use Dense to build dense artificial neural network (ANN)
* Apply the evaluate method on the testing data & plot the accuracy of ANN
* Plot the model loss progression