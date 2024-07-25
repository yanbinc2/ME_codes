#==========================================================
## ME Algorithm, 2024
#### Yan-Bin Chen (陳彥賓)(1)(2) yanbin@ntu.edu.tw <br>
#### Khong Loon Tiong (張孔綸)(1) khongloontiong@gmail.com <br>
#### Chen-Hsiang Yeang (楊振翔)(1) chyeang@stat.sinica.edu.tw
#### (1)Institute of Statistical Science, Academia Sinica, Taipei, Taiwan.
#### (2)Master Program in Statistics, CGE, National Taiwan University (NTU), Taipei, Taiwan.
#### July, 2024
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
          
          ...
          
          
          └── phase5

We provide a small dataset for users to experience the ME algorithm, which is a subset sampled from "PlantVillage". The dataset is named "ResNet18_PlantDisease_45K_Values_sampling.zip" and is located in the phase1/data/ directory. In other words, **the user doesn't need to run phase1 and can start from phase2**. We experimented with the ME algorithm on a GPU server consisting of an Intel® Xeon® CPU E5-2667 v4 3.20 GHz, 256 GB RAM, and an NVIDIA GPU Quadro P4000. The software we applied includes R 4.1.1, RStudio, the kknn package for R, Python 3.7.9, TensorFlow 2.1.0, and Matlab 9.10.0 (R2021a).

#
(b) File descriptions:

1. Phase1 (R language):
  
   'SpecClust.R' --> SpecClust.R takes the input of 43127 X 1000 of ResNet18 embeddings and outputs the result of clustering the images into 200 clusters.

2. Phase2 (Matlab):
  
   'generate_seedregions_package.m' --> the Matlab program of selecting seed regions from the data.

3. Phase3 (Python):
  
   'main_phase3.ipynb' --> the Python code for generating three CNN predicted results: 'original,' 'combinational,' and 'removal,' for the evaluation of four scores.

4. Phase4 (Matlab):
  
   'merge_seedregions_package.m' --> the Matlab program of merging seed regions according to three combinatorial classification outcomes (three mat files as shown above).

5. Phase5 (Python):
  
   'main_phase5.ipynb'n--> the Python code for merging and expanding seed regions.


#
(c) Execution procedures:

1. Phase1 (R language):
   
   Run 'SpecClust.R' to generate candidate seed regions. It outputs the files 'ResNet18_PlantDisease_45K_Spec200_sampling.csv' and 'ResNet18_PlantDisease_45K_Values_sampling.csv'.

2. Phase2 (Matlab):
   
   Copy the file 'ResNet18_PlantDisease_45K_Spec200_sampling.csv' and the embedded data file 'ResNet18_PlantDisease_45K_Values_sampling.csv' from phase1 into phase2/data/. Run 'generate_seedregions_example_package.m'. It will output three files: 'seedinds.txt', 'seedinds_neighborregions.txt' and 'bilabels.txt' in the phase2/data/.

3. Phase3 (Python):
   
   Copy the three output files 'seedinds.txt', 'seedinds_neighborregions.txt', and 'bilabels.txt' from phase2 to phase3/data/. Also, copy the output files 'ResNet18_PlantDisease_45K_Spec200_sampling.csv' and 'ResNet18_PlantDisease_45K_Values_sampling.csv' from phase1 to phase3/data/. Run 'main_phase3.ipynb'. It will output three mat files 'results_of_original.mat', 'results_of_combination.mat', and 'results_of_removal.mat' in the phase3/codes/.

4. Phase4 (Matlab):

   Copy the three mat files from phase3/codes/ into phase4/data/. Run 'merge_seedregions_package.m'. It will output one file "mergedseedclasslabels.txt" in the phase4/data/.

5. Phase5 (Python):
    
   Copy one output file from phase4 into phase5/data/. Run 'main_phase5.ipynb'. It will show the interpreted accuracy results as the python output.
