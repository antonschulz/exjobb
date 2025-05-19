# Dataset for classification of insects in event camera recordings

This dataset is part of the paper "Features for Classifying Insect Trajectories in Event Camera Recordings" by Regina Pohle-Fröhlich, Colin Gebler, Marc Böge, Tobias Bolten, Michael Glück and Kirsten Traynor.

This dataset contains data from the dataset of the WACV 2024 paper "Towards a Dynamic Vision Sensor-based Insect Camera Trap" (https://openaccess.thecvf.com/content/WACV2024/html/Gebauer_Towards_a_Dynamic_Vision_Sensor-Based_Insect_Camera_Trap_WACV_2024_paper.html) by Sebastian Thiele, Eike Jakob Gebauer, Pierre Ouvrard, Adrien Sicard and Benjamin Risse. The dataset can be found here: https://data-management.uni-muenster.de/datastore/download/10.17879/07938503301. It is licensed under CC BY 4.0 (https://spdx.org/licenses/CC-BY-4.0).
Our dataset includes data from only scene "3_m-h-h" of the Münster dataset. In our dataset all point cloud segments and flight trajectories from this scene start with "mu-3", either in the filename or the parent directory name. The data has been modified as described in our paper. This includes scaling of the t-dimension, noise reduction, sampling and normalization.


## Directories and files

**./full_trajectories/**

The subdirectory "full_trajectories" contains point clouds of complete, individual insect trajectories. Each subdirectory (e.g. "hn-bee-1") belongs to a scene. The first two characters identify the origin of the dvs recordings. "hn" and "mb" come from the Hochschule Niederrhein University of Applied Sciences (HSNR). "mu" comes from the University Münster. The following three characters (e.g. "bee") indicate which insect class was predominant in the scene. The numbers differentiate the scenes.

Each csv file is a full trajectory of an insect. The first number in the file name is the trajectory index within the scene. The following the characters (e.g. "bee") specify the class. The number following "pts" is the number of points/events of the trajectory. The number following "start" indicates the timestamp (in microseconds) of the first event of the trajectory within the recording, where 0 would be the start of the recording.
The csv files conatin the columns x, y, t and p. t is the unscaled timestamp in microseconds. p is the polarity. Maximum values for x and y are 1280 and 720.


**./fragmented_trajectories/**

The subdirectory "full_trajectories" contains fragmented trajectory point clouds. Each fragment has a duration of 100ms. The events/rows were shuffled.
The csv files conatin the columns x, y and t. All dimensions were normalized. That means all points are within a unit sphere (radius of 1 around the origin).

The file "train_test_split_7030.txt" in the subdirectories defines a 70:30 train/test-split. The first column is either "train" or "test". Second column specifies the class. Third column specifies the file (excluding the extension ".csv"). This split was used for the experiments in the paper.

"fragments.csv" contains additional information about each fragment. "options.json" contains parameter values that were used to fragment the trajectories.


**./fragmented_trajectories/100ms_4096pts_2048minpts_fps_sor_norm_shufflet/**

Each fragment was up- or downsampled to exactly 4096 points. Fragments with less than 2048 events were discarded. Fragments with at least 2048 events, but less than 4096 were upsampled to 4096 points. Fragments with more than 4096 points were downsampled with Farthest Point Sampling. Before sampling, Statistical Outlier Removal was applied. Finally each fragment was normalized.


**./fragmented_trajectories/100ms_2048pts_1024minpts_fps_norm_shufflet/**

Each fragment was up- or downsampled to exactly 2048 points. Fragments with less than 1024 events were discarded. Fragments with at least 1024 events, but less than 2048 were upsampled to 4096 points. Fragments with more than 2048 points were downsampled with Farthest Point Sampling. No noise reduction method was applied. Finally each fragment was normalized.




