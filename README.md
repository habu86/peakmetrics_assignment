Welcome and thank you for reviewing my submission for this assignmnent.

The Cluster_Analysis_Report.html file is the main product I am submitting for this assignment. Additional 
outputs can be found in the word_clouds folder where I have stored JPEGs with each generated cluster or Louvain
community's word clouds. Furthermore, their filenames correspond to automatically-generated labels using PoS tagging 
and frequency counts. Some are more meaningfully-descriptive than others ¯\_(ツ)_/¯

I have included Jupyter notebooks that should allow you to re-run my analysis. I apologize in
advance for the sparse documentation, should you choose to do this. I have also included an utilities file containing
a number of helper functions and wrappers used in the various notebooks. I have generated a requirements.txt
file listing all required packages. Please be advised that the pytorch and tensorflow workflows were run on a GPU
and the codebase assumes that one will be available. Lastly, due to the size of the datasets involved, the only data
artifacts I am including are those used by the very last portion of the workflow, namely the 
Cluster_Analysis_Report.ipynb notebook, which is used to generate the .html file containing the report itself.

Should you choose to re-run the analysis workflow, the order is as follows:
1. peakmetrics_data_prep_compact_code --> handles data intake and preparation for clustering
2. peakmetrics_document_clustering.ipynb --> uses outputs from step #1 to perform the actual clustering task
3. Airline Negative Sentiment Cluster Analysis --> uses outputs from steps #1 and #2 in order to facilitate analysis
4. Cluster_Analysis_Report ---> uses outputs from step #3 to generate the .html file for the final report

Be advised that the input data must already be present in the data prep notebook's directory. The assumed folder
structure is:

    home    
    |--> \news\*.parquet
    |--> \social\*.parquet
    |--> \blog\*.parquet

Running the entire workflow end-to-end is a multi-hour process and data structures have not been optimized to 
minimize memory usage; memory usage in the 30-40GB range has been observed during this workflow.

Please do not hesitate to reach out if you encounter issues with the codebase, or have any other questions.