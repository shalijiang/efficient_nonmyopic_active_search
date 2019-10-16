# Efficient Nonmyopic (batch) Active Search

This repo contains implentations of the active search policies in the following two papers:

[1]. Shali Jiang, Gustavo Malkomes, Geoff Converse, Alyssa Shofner, Benjamin Moseley, Roman Garnett. 
Efficient Nonmyopic Active Search. ICML 2017. http://proceedings.mlr.press/v70/jiang17d.html

[2]. Shali Jiang, Gustavo Malkomes, Matthew Abbott, Benjamin Moseley, Roman Garnett. 
Efficient Nonmyopic Batch Active Search. NeurIPS 2018. https://papers.nips.cc/paper/7387-efficient-nonmyopic-batch-active-search

[3]. Shali Jiang, Benjamin Moseley, Roman Garnett. 
Cost Effective Active Search. NeurIPS 2019.

# Video
A 3-minute video introducing efficient nonmyopic batch active search: https://www.youtube.com/watch?v=9y1HNY95LzY&feature=youtu.be

# How to use
Download the code,
and checkout "demo.m" line 1-4 to see how to add dependencies,
then run 

`>> demo` 

in Matlab to see how to use it. 

Change parameter settings to try different datasets and policies. 
In particular, change `which_setting` to switch between budgeted or cost effective settings. 

The code is partially tested on Ubuntu 18.04 with Matlab 2017b.

# Dependencies
Active learning toolbox: https://github.com/rmgarnett/active_learning.git 

Active search toolbox: https://github.com/rmgarnett/active_search.git 

For drug discovery datasets: https://github.com/rmgarnett/active_virtual_screening.git

GPML package to generate toy problem: http://www.gaussianprocess.org/gpml/code/matlab/doc/
