#==========================================================
## ME Algorithm, 2024
#### Yan-Bin Chen (陳彥賓) yanbin@ntu.edu.tw; <br>
#### Khong Loon Tiong (張孔綸) khongloontiong@gmail.com; <br>
#### Chen-Hsiang Yeang (楊振翔) chyeang@stat.sinica.edu.tw
#### Institute of Statistical Science, Academia Sinica, Taipei, Taiwan.
#### Master Program in Statistics, National Taiwan University, Taipei, Taiwan.
#### March, 2024
#==========================================================
#
(a) Abstract:

The ME five phases deployed here follow the following tree structure.

          main
          
          └── phase1
          
               └── codes

                    └── files
               
               └── data

                    └── files
              
          └── phase2
          
               └── codes

                    └── files
               
               └── data

                    └── files
          
          └── ...more

We provide a small toy dataset for users to experience the ME algorithm, which is a subset sampled from "PlantVillage". This toy dataset is an embedded dataset named "ResNet18_PlantDisease_45K_Values_sampling.zip", and is located in the phase1/data/ directory. We experimented with the ME algorithm on a GPU server consisting of an Intel® Xeon® CPU E5-2667 v4 3.20 GHz, 256 GB RAM, and an NVIDIA GPU Quadro P4000. The software we applied includes R 4.1.1, RStudio, the kknn package for R, Python 3.7.9, TensorFlow 2.1.0, and Matlab 9.10.0 (R2021a).

#
(b) File descriptions:

1.Phase1 (R language):

'SpecClust.R'

Function: SpecClust.R takes the input of 43127 X 1000 of ResNet18 embeddings and outputs the result of clustering the images into 200 clusters.


Inputs:
   

    "ResNet18_PlantDisease_45K_Values" - ResNet18 embedding values of PlantVillage samples.

    "ResNet18_PlantDisease_45K_Labels" - Labels of PlantVillage samples


Outputs:

    "ResNet18_PlantDisease_45K_Spec200" - Result of clustering appended as a new column to the original  "ResNet18_PlantDisease_45K_Labels" file.
   

2.Phase2 (Matlab):

'generate_seedregions_package.m'

Function: the Matlab program of selecting seed regions from the data.
   
Inputs: Details for each parameter are described as follows.
   
    data: The data undergone a fixed embedding.

    augdata: The augmented data generated by manipulating the original data (rotation, translation, etc).  They are used to identify the informative markers for specific regions.

    tsnedata: The t-SNE projection of the embedded data.
   
    regionneighborlabels: The membership vector of region labels for all images.
   
    filter: A flag indicating whether the program incorporates geometric information to filter images (filter=1), incorporates marker information to filter images (filter=2), or does not filter images (filter=0).
   
    numthre: This hyper parameter is used when filter=1.  The threshold of determining whether a region is a valid region.  If the number of non-periphery members of the region < numthre, then the region is invalid.  numthre is set to 100 when most regions generated by spectral clustering have about 200-500 members.
   
    fracthre: This hyper parameter is used when filter=1.  The threshold of determining whether a region is a valid region.  If the number of non-periphery members of the region < fracthre, then the region is invalid.  fracthre is set to be in the range of 0.8-0.9.
   
    topthre: This hyper parameter is used when filter=2.  The threshold of determining top rank images of a marker.  For each image, convert the embedding values of all markers into ranks.  The markers within the rank of topthre have high activation outputs for that image.  topthre is set to 20.
   
    nimagesthre: This hyper parameter is used when filter=2.  The threshold of determining the number of top-ranking images of a marker.  If a marker has >= nimagesthre top-ranking images, then select that marker for further consideration.  For CIFAR10 data, there are totally 60000 images.  nimagesthre is set to 1000.
   
    mthre: This hyper parameter is used when filter=2.  The threshold for determining whether a marker is a valid marker in a region.  For each marker and each region, sort the members by marker rank values in an increasing order, and subdivide the members into lower and upper subgroups.  Evaluate the mean rank value of the lower subgroup over the original and multiple augmented datasets.  If the mean rank value <= mthre, then the marker is valid for that region.  This means that at least one half of the region members have high activation values (low average rank values) in original and augmented datasets.  For the ResNet50 and VGG16 embeddings, there are 1000 markers.  Set mthre to be in the range of 100 and 200 (top 10-20 percent).
   
    lwdthre and uppthre: These hyper parameters are used when filter=2.  The thresholds for determining the separation markers of regions.  A region may comprise members with distinct underlying class labels.  Some markers may be able to separate the members with the dominant class labels and the non-dominant class labels.  These hyper parameters help identifying the separation markers of a region.  For each region and each marker, sort and subdivide member images as in the description of mthre.  Calculate the mean rank values of the lower and upper groups.  If the lower group mean value <= lwdthre and the higher group mean value >= uppthre, then this marker is a separation marker of the region.  Set lwdthre to be in the range of 100 and 150, and uppthre to be in the range of 250 and 300.

    nmarkersthre: This hyper parameter is used when filter=2.  Threshold for judging whether a region has too mixed class label composition to be considered.  If a region has >= nmarkersthre, then discard it as a candidate seed region.  Set nmarkersthre to be 5.
   
    rankthre and rankratiothre: These hyper parameters are used when filter=2.  Thresholds for determining valid separation markers of each region.  For each region and each of its separation marker, count the fraction of images whose rank values <= rankthre.  If this fraction < rankratiothre, then set this separation marker valid.  This procedure filters out the separation markers that have generally high activation values.  rankthre is set to 100 (the embedding has 1000 markers).  rankratiothre is set to 0.3.

    overlapthre: This hyper parameter is used when filter=2.  For each region, if it contains valid separation markers, then for each valid separation marker partition the members into two subsets.  If the overlap ratio exceeds overlapthre in each partition, then consider these partitions consistent.  Build a consensus partition to subdivide the members into two subsets.  Set overlapthre to 0.75.

    sizethre and disparitythre: These hyper parameters are used when filter=2.  Thresholds for determing whether the consistent partition (if ever exists) should be undertaken.  Count the size of the two partitioned subsets.  If the two subsets are too small (the larger size < sizethre) or the two subsets have similar sizes (the ratio of the larger size and the smaller size < disparitythre), then do not partition the region members.  Set sizethre to 100 and disparitythre to 1.2.

    highthre: This hyper parameter is used when filter=2.  Threshold for determining the markers for each region.  For each region and each selected marker, count the fraction of members whose rank values <= topthre.  If this fraction >= highthre, then assign the selected marker as a marker of the region.  Set highthre=0.3.

    dthre: This hyper parameter is used when filter=2.  Threshold for determining informative markers.  For each marker, extract the regions which have high fraction of images with high ranking values (ranking values <= topthre, fraction of images with high ranking values >= highthre).  Examine how spread out the selected regions are.  For each pair of selected regions, obtain their average distance in terms of its rank among pairwise distances.  If the median value of the rank distances <= dthre, then label the marker as an informative marker.  Set dthre=25.

    nnbsthre: This hyper parameter is used when filter=2.  The number of the nearest neighboring regions to be examined.  Set nnbsthre=10.

    sharedthre: This hyper parameter is used when filter=2.  Threshold for determining valid neighboring regions.  For each region, obtain the nnbsthre nearest neighboring regions.  For each neighboring region, count the fraction of shared region informative markers among the union of their region informative markers.  If this fraction >= sharedthre, then this neighboring region is a valid neighboring region.  Set sharedthre=0.2.

    nvalidnbsthre: This hyper parameter is used when filter=2.  Threshold for determining valid regions.  If a region has >= nvalidnbsthre valid neighboring regions, then it is a valid region.  Set nvalidnbsthre=5.

    rthre: This hyper parameter is used to find the first two seed regions.  The first two seed regions should be far apart but avoid selecting outlier regions which are distant from other regions.  For each pair of regions (say 1 and 2), count the number of regions closer to either one of them (1 or 2).  Calculate the ratio between the larger and the smaller of the numbers of regions.  This ratio quantifies the balancedness of the numbers of regions attracted to either member of the pair.  To select the first two seed regions, we first sort region pairs by their average distances in an increasing order, then select the first region pair where (1) both regions in the pair are valid regions, (2) the balancedness ratio <= rthre.  rthre is set to be in the range between 1 and 1.5.

    regionpairDmode: This hyper parameter is used in calculating the average pairwise distances between regions. If regionpairDmode=0, then the program uses all the members of the regions to evaluate the average distances.  If regionpairDmode=1, then the program uses the non-periphery members of the regions to evaluate the average distances.  Set regionpairDmode=0 for CIFAR10 and regionpairDmode=1 for other datasets.

    maxnseeds: The number of iterations for incrementally adding candidate seed regions.  Set maxnseeds to 40 for MNIST and CIFAR10 and 22 for other datasets.

    nclassesthre, ratiothre and foldthre: These hyper parameters are used to select the candidate seed regions.  For an existing seed region i and a new candidate seed region r, evaluate r^{conf}_{i}(r), the size difference between the incompatible and compatible prediction sets restricted to class i and normalized by the size of the compatible prediction set (equation 6 in the manuscript).  If the number of existing seed regions <= nclassesthre, then consider the region r as a candidate of the next seed region if the r^{conf}_{i}(r) value <= ratiothre.  If the number of existing seed regions > nclassesthre, then consider the region r as a candidate of the next seed region if the r^{conf}_{i}(r) value <= ratiothre and <= the minimum value of (r^{conf}_{i}(r)*foldthre).  Set nclassesthre=4, ratiothre to 10.0 and foldthre to 10.0.

    diffratiothre: This hyper parameter is used to select the candidate seed regions.  Find the new seed region r which has the max min distance from existing seed regions (equation 2).  For an existing seed region i, replace seed region i with the new region r in the training data of the predictor, evaluate n^{diff}_{i}(r) the number of images where the original and modified predictors give different prediction labels.  Find the minimum of the n^{diff}_{i}(r) scores over all existing seed regions i.  If this minimum value >= (nimages*diffratiothre), then include the selected new seed region in the existing seed regions.  Set diffratiothre in the range of 0.02 and 0.05.


  
Outputs:

    'seedinds.txt'

    'seedinds_neighborregions.txt'

    'bilabels.txt'


3.Phase3 (Python):

'main_phase3.ipynb'

Function: the Python code for generating three CNN predicted results: 'original,' 'combinational,' and 'removal,' for the evaluation of four scores.

Inputs: the files specified by the paths are the input data. The input files are demonstrated for the instructional guidance. Users may input their own files based on their applications.
   
    PATH1='../data/seedinds.txt'  -->  the seed regions ('seedinds') given by the phase 2.
  
    PATH2='../data/bilabels.txt'  --> the data file to indicate which labels ('bilabels') are effective given by phase 2.
  
    PATH3='../data/seedinds_neighborregions.txt'  --> to specify the neighboring regions of seed regions.
   
    PATH4='../data/ResNet18_PlantDisease_45K_Spec200.csv'  --> the seed region index.
   
    PATH5='../data/ResNet18_PlantDisease_45K_Values.csv'  --> the embedded data.
    
    Followings are parameters:
  
    TRIALS: the number of trials for the CNN. Usually, set to 1.
   
    timestr: the prompt for the output file.

    REG_COLUMN: column name for region index.

    RAW_2D_DATA: 2D data is True; 1D data is False

  
Outputs: output three CNN predicted files for the four scores evaluation, as follows:
   
    'results_of_original.mat'  --> the predicted results of original CNN.

    'results_of_combination.mat'  --> the predicted results of combinatorial CNN. 

    'results_of_removal.mat'  --> the predicted results of removal CNN.


4.Phase4 (Matlab):

'merge_seedregions_package.m'

Function: the Matlab program of merging seed regions according to three combinatorial classification outcomes (.mat files).

Inputs: information about regions and seed regions, prediction outcomes, and thresholds. Details for each parameter are described as follows.

    nseeds: Number of seed regions.
   
    seedinds: Seed region indices.
   
    result_for_original: An ntrials*nimages matrix reporting the CNN predictions of all images in multiple trials.  The training data is the seed region members.  Each seed region comprises each class.

    prob_for_original: An ntrials*nimages*nclasses tensor reporting the CNN final layer neuron output values of all images in multiple trials.

    combination_pairs: All pairs of seed regions.
   
    result_for_merge: An npairs*nimages matrix reporting the CNN predictions for merging each pair of seed regions as a training class.
   
    result_for_removal: An nseeds*nimages matrix reporting the CNN predictions for removing each seed region from the training data.
   
    data: The embedded data.
   
    regiontraininglabels: The region labels of all images.  If an image is a periphery member, then set the label to 0.
   
    confusionratiothre: Threshold on the confusion ratio of prediction outcomes to add extra links in the graph of seed regions.  For a pair of seed regions i and j, denote C(i,j) the number of images which belong to seed region i but are assigned to seed region j by CNN prediction.  If C(i,j) >= (C(i,i)*confusionratiothre), then add a link between nodes i and j.  Set confusionratiothre to 0.1.

    usemarker: A binary variable indicating whether the program uses information about informative markers.

    npartitionnodes: This variable is used when usemarker=1.  Number of partition nodes in the decision tree of the markers.  A partition node is a node in the marker decision tree where its two children have very distinct data patterns.

    partitionnodes: This variable is used when usemarker=1.  Indices of partition nodes in the decision tree of the markers.  Each partition node corresponds to a group of non-overlapped markers.

    seedmarkergroupindicators: This variable is used when usemarker=1.  An (nseeds*npartitionnodes) matrix reporting the quantized binary values of each seed region on the marker group of the corresponding partition node.

    nprednumthre: Threshold on the number of valid and consensus predictions assigned to a seed region.  A seed region is discarded if this number < nprednumthre.  Set nprednumthre=100.

    pvalthre: Threshold on the rank-sum test p-values of the leakage scores.  If the leakage score of a pair of seed regions >= pvalthre, then these two seed regions should not be merged together.  Set pvalthre=0.2.

    ntoprankthre: For each seed region, threshold on the number of top ranking partner seed regions.  Construct a graph of the top-ranking partners of seed regions in terms of reducedrankscores scores.  Set ntoprankthre=3.

    importthre: A binary variable indicating whether the threshold values on the pairwise scores are determined from the distributions of the scores (importthre=0) or imported from inputs (importthre=1).  Set importthre=0 for MNIST data and importthre=1 for other datasets.

    quantilethre: This hyper parameter is used when importthre=0.  Sort the pairwise scores of cocontributions and replacement scores in an increasing order and the leakage logrank p-values in a decreasing order.  Calculate the quantilethre values of these three pairwise scores.  Use these values as cocontributionthre, replacementratiothre, and localranksumpvalthre respectively.  Set quantilethre in the range between 0.7 and 0.9.

    cocontributionthre, replacementratiothre, localranksumpvalthre: These hyper parameters are used when importthre=1.  Thresholds of determining whether each pair of regions are mergeable according to each type of pairwise score.  They are explicitly imported when importthre=1 and determined from the distributions of pairwise scores when importthre=0.

    sizethre: Threshold on the min size of subgraphs undergoing spectral clustering. Start with the whole graph of symmetrized reducedsumscores and iteratively incur spectral clustering to generate binary partitions.  If a subgraph size <= sizethre, then stop partitioning.  Set sizethre to 7.

    smallclustersizethre: Threshold of the cluster size for considering outlier clusters.  Apply k-means (k=2) clustering to the projected subgraph nodes on 2D space according to spectral clustering.  Get the large and small clusters.  If the small cluster size <= smallclustersizethre, then consider the small cluster as an outlier cluster.  Set smallclustersizethre to 5.

    jumpfoldthre: Threshold of the difference between intra and inter cluster distances to determine outlier clusters.  Find the node pairs within the small cluster and between the small and large clusters.  Denote val1 the max of intra cluster distances and val2 the min of inter cluster distances.  If val2 >= (val1*jumpfoldthre), then mark the small cluster as an outlier cluster.  Set jumpfoldthre=5.

    gapfoldthre: Threshold on the difference between the first and second biggest gaps to decide whether to split the subgraph.  Project the subgraph nodes on the first dimension of spectral clustering, and calculate the first and second biggest gaps between the projected values.  If the first biggest gap >= (the second biggest gap*gapfoldthre), then split the subgraph.  Set gapfoldthre=2.0.

    smallratiothre: Threshold on the ratio between the minimum distance of positive and negative sides of projected values and max distance of projected values.  Denote val1 the max distance between subgraph node projected values on the first dimension, and val2 the min distance between the projected values on positive and negative sides.  If the ratio val2/val1 <= smallratiothre, then split the subgraph.  Set smallratiothre to 0.1.

    pdiffthre and medvalthre: These hyper parameters are used when usemarker=1.  Threshold on pdiff scores to determine the markers specific to each seed region.  For each marker and each pair of seed regions, evaluate the distributions of marker values on the members of these two regions, and calculate the pdiff scores of the distributions.  For each marker and each seed region, obtain the minimum pdiff score over all other seed regions.  If the minimum pdiff score >= pdiffthre, then calculate the median value of the entries of the (marker,seed region) combination.  If the median value >= medvalthre, then assign the marker to the seed region.  Set pdiffthre=0.7 and medvalthre=7.0.

    ninformativemarkersthre: This hyper parameters is used when usemarker=1.  Threshold on the number of informative markers to determine singleton regions.  If a seed region has >= ninformativemarkersthre informative markers, then the seed region is a singleton.  Set ninformativemarkersthre=5.

    ninformativemarkersthre2: This hyper parameter is used when usemarker=1.  Another threshold on the number of informative markers to determine singleton regions.  Use another criterion to determine the marker genes of each seed region.  If a seed region has >= ninformativemarkersthre2 informative markers, then the seed region is a singleton.  Set ninformativemarkersthre2=40.

    pdiffthre2: This hyper parameters is used when usemarker=1.  Threshold on pdiff scores to repartition each component generated from the first partition.  It is used in the same way as pdiffthre.  Set pdiffthre2=0.5.
 
    cntdiffthre: This hyper parameter is used when usemarker=1.  Threshold on how a seed region is distinct from other seed regions to qualify as isolated.  For each seed region, count the number of informative markers, and sort them in a decreasing order.  If the first seed region has >= cntdiffthre more informative markers than the second seed regions, and the number of informative markes in the second seed region < 100, then label the first seed region isolated.  Set cntdiffthre=8.

    reducedrankscorethre: Threshold on reducedrankscores to determine mergeable relations.  For each pair of seed regions, if the reducedrankscore <= reducedrankscorethre, then mark the seed region pair mergeable.  Set reducedrankscorethre=10.
 
    cssizethre: Threshold on the number of non-singleton members of a component.  If the number of non-singleton members > 0 and <= cssizethre, then treat it as the merged class.  Set cssizethre=3.


Outputs:
   
    'mergedseedclasslabels.txt'

5.Phase5 (Python):

'main_phase5.ipynb'

Function: the Python code for merging and expanding seed regions.

Inputs: the files specified by the paths are the input data. The input files are demonstrated for the instructional guidance. Users may input their own files based on their applications.
    
    PATH4='../../phase3/data/ResNet18_PlantDisease_45K_Spec200.csv'  --> the data file to specify the entire region index.
  
    PATH5='../../phase3/data/embedded_data.pickle'  --> the true labels for the accuracy evaluation.
  
    PATH6='../data/mergedseedclasslabels.txt' --> the merged results given by the phase 4.
  
    PATH7='../../phase3/data/region_for_phase5.pickle'  --> the initial conditions.

    
    
    Followings are parameters:

    MNIST: for t-SNE embedded data in MNIST. True: MNIST; False: not MNIST
    
    NUM_CASE: to indicate how many cases in the merged seed regions outcomes.
   
    INTE_bool: the switch to specify the intergration netwrok mode or single network mode.
    
    SAVE_bool: to save the results.
    
    ITE_FROM: the start point of the loop. This is for the special case or debugging.
    
    REG_COLUMN: column name for region index.

    RAW_2D_DATA: 2D data is True; 1D data is False
    
    interpret_path: the path of the output file
    
    AMOUNT_ITE: Number of iterations
  
Outputs: output the accuracy tables.

    'accu_history.csv'  --> accuracy tables.

#
(c) Execution procedures:
1. Phase1: Run 'SpecClust.R' to generate candidate seed regions. It outputs a file 'ResNet18_PlantDisease_45K_Spec200.csv' in phase1/data/.
2. Phase 2: Copy the output file 'ResNet18_PlantDisease_45K_Spec200.csv' from phase1 and the embedded data file 'ResNet18_PlantDisease_45K_Values.csv' into phase2/data/. The hyperparameter file 'PlantVillage_generate_seedregions_params.txt' has already been set and stored in phase2/data/. Run 'generate_seedregions_example_package.m'. It will output three files: 'seedinds.txt', 'seedinds_neighborregions.txt' and 'bilabels.txt'.
3. Phase3. Copy the three output files 'seedinds.txt', 'seedinds_neighborregions.txt', and 'bilabels.txt' from phase2 to phase3/data/. Also, copy the output file 'ResNet18_PlantDisease_45K_Spec200.csv' from phase1 to phase3/data/. It will convert 'ResNet18_PlantDisease_45K_Spec200.csv' into a pickle file 'embedded_data.pickle' for the convenience, and will place it into phase3/data/. Run 'main_phase3.ipynb'. It will output several immediate pickle files. Finally, the three mat files 'results_of_original.mat', 'results_of_combination.mat', and 'results_of_removal.mat' will be generated in the phase3/codes/.
4. Phase4. Copy the three mat files from phase3/codes/ into phase4/data/. The hyperparameter file 'Plant_merge_seedregions_params.txt' has been stored in phase4/data/. Run 'merge_seedregions_package.m'. It will output one file "mergedseedclasslabels.txt".
5. Phase5. Copy one output file from phase4 into phase5/data/. Run 'main_phase5.ipynb' to obtain the clustering results 'accu_history.csv' in phase5/codes/. Furthermore, it will show the interpreted accuracy results as the python output.
