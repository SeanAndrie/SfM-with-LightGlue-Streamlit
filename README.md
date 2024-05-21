# Structure-from-Motion with LightGlue
This Github repository features the code for a Streamlit application for performing Structure-from-Motion (SfM) reconstructions using the LightGlue feature matcher. The application allows users to upload their own image datasets, select feature matchers, and generate 3D point cloud reconstructions.

## Deploy on Google Colab for GPU
Structure-from-Motion is quite computationally intensive, and so, it is recommended that you leverage Google Colab's free GPU to boost the application. You can do this by:
1. Cloning the `colab-sfm-w-lightglue` python notebook into [Colab](https://colab.research.google.com/) and run all the cells.
2. Once all cells are running, you will see a link in the last cell that looks something like this: https://five-parents-crash.loca.lt.
3. Navigate to the local tunnel website and use the tunnel password in the colab notebook.
