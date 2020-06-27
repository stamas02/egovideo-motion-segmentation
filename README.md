# egovideo-motion-segmentation

The aim is to segment a video based on motion clues. Each video segment 
should contain limited amount of motion. The maximum amount of motion allowed
is controlled by a threshold parameter. 

# Optical flow
We follow [1] and define a 10x5 grid on the image. Each grid cell we compute
optical flow using a Lucas-Kanade [2] feature tracker implemented in the
opencv package (cv2.goodFeaturesToTrack). The average magnitude is then taken
and a singe displacement vector is assigned a grid cell. 

# Formal
We divide the image into a grid WxH (10x5 but can be changed with a parameter.)
and we compute the displacements: 
<img src="https://latex.codecogs.com/gif.latex?d_t(i,j) = (d_t^x(i,j), d_t^y(i,j))" />
at time t where i and j identify a grid cell. 

To calculate the total magnitude of the movement taken place since t=0 we
take the mean of the cumulated magnitude of the displacement vectors
 at time t:

<img src="https://latex.codecogs.com/gif.latex?M_t = m(d_t(i,j))+m(d_{t-1}(i,j))" />

Where m if a function calculating the magnitude of the vector and 
<img src="https://latex.codecogs.com/gif.latex?M_t" />
is the cumulated Magnitude since t=0. when 
<img src="https://latex.codecogs.com/gif.latex?M_t>T" /> We reset t=0
and cut the video. We tried the algorithm with T=156

# Usage
```
python segment.py -v PATH/TO/YOUR/VIDEO.MP4 -o PATH/TO/YOUR/OUTPUT/DIRECTORY -g 10 5 -t 156
```
Remember to create the output directory before running the script.

You can pass the following additional arguments to segment.py:

| arguments:           | Description |
| ----------------              | --- |
| -h, --help                            | show this help message and exit |
| --video VIDEO, -v VIDEO               | path to the video file.|
| --threshold THRESHOLD, -t THRESHOLD   | The threshold parameter T for segmentation |
| --grid-size GRID_SIZE [GRID_SIZE ...], -g GRID_SIZE [GRID_SIZE ...]|   A touple representing the ncols and nrows of the grid|
| --output-dir OUTPUT_DIR, -o OUTPUT_DIR| folder where the segmentation and plots are saved |

# Results
Result files created by segment.py:

| File:           | Description |
| ----------------              | --- |
|segmentation.pickle|A pandas dataframe with columns: ["Frame", "Name", "Magnitude", "Threshold", "Class" ]|
|segmentation_stat.csv|Segmentation statistic|
|segmentation_plot.(html,svg)|A line plot where the x axis represents the frames and the y represents <img src="https://latex.codecogs.com/gif.latex?M_t" /> |

## Example result
![Alt text](log/segmentation_plot.svg)
<img src="log/segmentation_plot.svg">

# References
[1] Yair Poleg, Ariel Ephrat, Shmuel Peleg, & Chetan Arora (2016). Compact CNN for Indexing Egocentric Videos. In WACV.

[2] Lucas, B., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. In IJCAI (pp. 674-679).

# Bibs
[1]
```
@inproceedings{poleg_cvpr14_egoseg,
  title     = {Temporal Segmentation of Egocentric Videos},
  author    = {Yair Poleg and Chetan Arora and Shmuel Peleg},
  year      = {2014},
  booktitle = {CVPR}
}
```
[2]
```
@INPROCEEDINGS{lucas_kanade,
author={Lucas, Bruce D and Kanade, Takeo},
booktitle={IJCAI},
title={An iterative image registration technique with an application to stereo vision},
year={1981},
pages={674-679},
}
```