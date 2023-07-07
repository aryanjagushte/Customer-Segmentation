# Customer Segmentation
This project aims to perform customer segmentation using the K-means clustering algorithm.
The dataset used for this analysis is the Mall Customers dataset, which contains information about customers' annual income and spending score.

# DATASET
The dataset (Mall_Customers.csv) consists of the following columns:

*CustomerID: Unique identifier for each customer
*Gender: Gender of the customer (Male/Female)
*Age: Age of the customer
*Annual Income (k$): Annual income of the customer in thousands of dollars
*Spending Score (1-100): Score assigned to the customer based on their spending habits

# STEPS
1) Loading the Dataset: The dataset is loaded using the pandas library.

2) Exploratory Data Analysis: Basic exploration of the dataset, such as checking the shape, displaying the first few rows, and gathering information about the columns.

3) Data Preprocessing: Extracting the relevant columns for clustering and converting them to a numpy array.

4) Finding Optimal Number of Clusters: Applying the Elbow method to determine the optimal number of clusters.

5) Training the Model: Creating a K-means clustering model with the determined number of clusters and fitting it to the data.

6) Visualizing the Clusters: Plotting the clusters and centroids to visualize the customer segments based on their annual income and spending score.

# DEPENDENCIES
This project requires the following Python libraries:
pandas
numpy
matplotlib
seaborn
sklearn
Please make sure to install the necessary dependencies before running the code

# USAGE
Clone the repository and navigate to the project directory.

Make sure the dataset file (Mall_Customers.csv) is placed in the same directory.

Run the Python script in any Python IDE or Jupyter Notebook.

The output will include the visualization of customer segments.

Feel free to modify and experiment with the code to explore further insights or apply it to different datasets.

**Note: This project is for educational purposes and serves as a starting point for customer segmentation analysis.
        Further enhancements and improvements can be made to refine the results and extend the functionality
