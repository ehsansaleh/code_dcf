# Firth Bias Reduction in Few-shot Distribution Calibration
This sub-directory contains the experiments in which Firth bias reduction from our paper [On the Importance of Firth Bias Reduction in Few-Shot Classification](https://openreview.net/pdf?id=DNRADop4ksB) is applied to the recent State-of-the-Art [Distribution Calibration (DC)](https://openreview.net/pdf?id=JWOiYxMG92s) method in few-shot classification. This is one of the three code repositories of our paper, and is a sub-module of the the main ["Firth Bias Reduction in Few-Shot Learning" repository](https://github.com/ehsansaleh/firth_bias_reduction). 

For a concise and informal description of our work, check out our paper's website: [https://ehsansaleh.github.io/firthfsl](https://ehsansaleh.github.io/firthfsl)

<details open>
<summary><h2>Check Out Our GPU Implementation of the DC Method</h2></summary>
 
  * **If you use our GPU implementation of the DC method, please cite our paper ["On the Importance of Firth Bias Reduction in Few-Shot Classification, ICLR 2022"](#references).**
  * **We have re-implemented the DC method so that the few-shot classification runs entirely on GPU.**
  * **The original DC implementation utilized the scikit-learn library for training logistic classifiers in few-shot evaluation.**
    * At the time of publication, the default optimizer for logistic classification in scikit-learn was L-BFGS.
    * This is why our implementation utilizes the L-BFGS optimizer in Pytorch.
  * **Our version is faster than the original CPU version available at the [Few-shot DC github repository](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration).**
    * The DC method augments 750 artificial samples when training logistic classifiers, which is much more computationally demanding than typical few-shot evaluations.
    * Sampling the 750 normal samples from the calibrated distribution is done using the `numpy` library. 
    * Note that a schur decomposition (or a matrix inversion) of the 640-by-640 calibrated covariance matrix must be performed to sample from the calibrated normal distribution, which can be quite slow on CPU.
    * Considering that 10,000 tasks need to be repeated, the original implementation can require a lot of CPU resources for evaluation.
    * This is why we re-implemented the DC method (including the training of the classifier and the sampling of the calibrated distribution) on GPU. 
    * We did our best to reproduce the original paper's numbers; you can see our baseline 5-ways numbers on mini-ImageNet and tiered-ImageNet in the tables below.
  
</details>

<details open>
<summary><h2>Final Evaluations and Tables</h2></summary>
 
Following our step-by-step guide, you can produce the following tables.
  + <details open>
    <summary><strong>Cross-Domain Experiments and Firth Bias Reduction</strong></summary>

    <div align="center">
    <table><thead><tr><th rowspan="2">Way</th><th rowspan="2">Shot</th><th colspan="3">miniImagenet =&gt; CUB</th><th colspan="3">tieredImagenet =&gt; CUB</th></tr><tr><th>      Before     </th><th>After</th><th>Boost</th><th>Before</th><th>After</th><th>Boost</th></tr></thead><tbody><tr><td>10</td><td>1</td><td>37.14±0.12</td><td>37.41±0.12</td><td>0.27±0.03</td><td>64.36±0.16</td><td>64.52±0.16</td><td>0.15±0.03</td></tr><tr><td>10</td><td>5</td><td>59.77±0.12</td><td>60.77±0.12</td><td>1.00±0.04</td><td>86.23±0.10</td><td>86.66±0.09</td><td>0.43±0.03</td></tr><tr><td>15</td><td>1</td><td>30.22±0.09</td><td>30.37±0.09</td><td>0.15±0.03</td><td>57.73±0.13</td><td>57.73±0.13</td><td>0.00±0.00</td></tr><tr><td>15</td><td>5</td><td>52.73±0.09</td><td>53.84±0.09</td><td>1.11±0.03</td><td>82.16±0.09</td><td>83.05±0.08</td><td>0.89±0.03</td></tr></tbody></table>
    </div>
 
    + <details>
      <summary><strong>More Information</strong></summary>

      * This table was generated at [`tables/crossdomain.csv`](./tables/crossdomain.csv).
      * The relavant configs can be found at the [`configs/1_mini2CUB`](./configs/1_mini2CUB) and [`configs/2_tiered2CUB`](./configs/2_tiered2CUB) directories.
     </details>
 
   </details>

  + <details open>
    <summary><strong>Firth Bias Reduction Improvements on tiered-ImageNet</strong></summary>
 
    <div align="center">
    <table><thead><tr><th rowspan="2">Way</th><th rowspan="2">Shot</th><th colspan="3">No Artificial Samples </th><th colspan="3">750-Artificial Samples </th></tr><tr><th>Before</th><th>After</th><th>Boost</th><th>Before</th><th>After</th><th>Boost</th></tr></thead><tbody><tr><td>10</td><td>1</td><td>59.44±0.16</td><td>60.07±0.16</td><td>0.63±0.03</td><td>61.85±0.16</td><td>61.90±0.16</td><td>0.05±0.02</td></tr><tr><td>10</td><td>5</td><td>80.52±0.12</td><td>80.85±0.12</td><td>0.33±0.03</td><td>79.66±0.12</td><td>80.07±0.12</td><td>0.42±0.04</td></tr><tr><td>15</td><td>1</td><td>52.68±0.13</td><td>53.35±0.13</td><td>0.67±0.03</td><td>54.57±0.13</td><td>54.62±0.13</td><td>0.05±0.02</td></tr><tr><td>15</td><td>5</td><td>75.17±0.10</td><td>75.64±0.10</td><td>0.46±0.03</td><td>73.88±0.11</td><td>74.40±0.11</td><td>0.53±0.04</td></tr></tbody></table>
    </div>
    
    + <details>
      <summary><strong>More Information</strong></summary>
 
      * This table was generated at [`tables/tiered.csv`](./tables/tiered.csv).
      * The relavant configs for this table can be found at the [`configs/3_tiered2tiered`](./configs/3_tiered2tiered) directory.
      </details>
 
    </details>

  + <details open>
    <summary><strong>Baseline 5-way Accuracy on mini-ImageNet and tiered-ImageNet</strong></summary>
 
    <div align="center">

    | Way  	| Shot 	| miniImagenet   	| tieredImagenet 	|
    |------	|------	|----------------	|----------------	|
    | 5    	| 1    	| 67.89±0.20 	| 74.73±0.22 	|
    | 5    	| 5    	| 83.01±0.13 	| 88.34±0.14 	|
    </div>
 
    + <details>
      <summary><strong>More Information</strong></summary>
 
      * This table was generated at [`tables/5ways_mini_tiered.csv`](./tables/5ways_mini_tiered.csv).
      * The relavant configs for this table can be found at the [`configs/4_5ways`](./configs/4_5ways) directory.
      * For mini-imagenet, the results are either in a statistical tie or slightly better than the values reported in the [Few-shot Distribution Calibration paper](https://openreview.net/pdf?id=JWOiYxMG92s).
      * For tiered-imagenet, we could not reproduce the DC paper's numbers. While we [reported this mismatch](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration/issues/30) to the DC paper authors, the issue has not been resolved yet at the time of writing. (**Update**: The DC authors closed the entire issues section on their github repository, so we may never get the answer).
        * Apparently, the tiered-ImageNet results were added to the DC paper in the rebuttals period.
        * Since the original code was not updated after the rebuttals, the original DC repository cannot run the tiered-Imagenet experiments as-is, and there are a few missing lines of code which may be the key to this difference.
        * As soon as we hear back from the DC authors, we will try to update our code accordingly.
      </details>
    </details>

</details>

<details>
<summary><h2>Quick Q&A Rounds</h2></summary>

1. **Question**: Give me a quick-starter code to start reproducing the paper trainings on a GPU?
   ```bash
   git clone https://github.com/ehsansaleh/firth_bias_reduction.git
   cd ./firth_bias_reduction/code_dcf
   ./features/download.sh
   ./main.sh
   ```
---------
2. **Question**: Give me a simple python command to run?
   ```bash
   python main.py --device "cuda:0" --configid "1_mini2CUB/5s10w_0aug" 
   ```

    <details>
    <summary><strong>More Information</strong></summary> 
    
      * This will run the configuration specifed at [`./configs/1_mini2CUB/5s10w_0aug.json`](./configs/1_mini2CUB/5s10w_0aug.json).
      * This will store the generated outputs periodically at `./results/1_mini2CUB/5s10w_0aug.csv`.
     </details>

---------
3. **Question**: How can I reproduce the paper tables?

   ```bash
   make summary
   make figures
   make tables
   ```
  
   <details>
   <summary><strong>More Information</strong></summary>
   
   1. If you have run new classifier trainings by either `./main.sh` or `python main.py`, then run `make summary`. Otherwise, skip this step. This command will collect the csv files from the `./results` directory, and process them into a single summarized file at [`./summary/test2test.csv`](./summary/test2test.csv).
   2. Run `make tables` in case you're interested about the raw numbers at the [`./tables`](./tables) directory.
   
   </details>


---------
4. **Question**: I have my own code and I do not want to use your code. How can I apply the Firth bias reduction to my own loss?

   ```python
   ce_loss = nn.CrossEntropyLoss()
   ce_term = ce_loss(logits, target)
  
   log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
   firth_term = -log_probs.mean()
  
   loss = ce_term + lam * firth_term
   loss.backward()
   ```
   
   * Alternatively, you can use the `label_smoothing` keyword argument in [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). 
  
   * Remember that this Firth formulation is only true for 1-layer logistic and cosine classifiers. For more complex networks, the FIM's log-determinant must be worked out.
  
---------
   <details>
   <summary>4. <strong>Question:</strong> You seem to have too many directories, files, and a fancy structure. Explain the whole thing as simple as possible?</summary>
   
   
     
   ```
                    ./main.sh or
                   python main.py                  make summary                  make tables   
   configs/*.json ================> results/*.csv =============> summary/*.csv ===============> tables/*
                         /\                                                            
                         ||
            (below is    ||
             optional    ||
              parts)     ||
                         ||                        python save_features.py                   
                         ======= features/*.pkl <========================== checkpoints/*.tar
                                       /\
                                       ||
                                       ||
                                    Datasets/*
   ```

   The top horizontal line is the important one for our work.

   </details>
  
---------
   <details>
   <summary>5. <strong>Question:</strong> What are the python environment package requirements?</summary>
   
   * We ran the code using `python 3.8`.
     
   * The classifier training code mainly needs `numpy`, `torch`, `torchvision`, and `pandas`.
     
   * For generating the figures, you also need `matplotlib`, `seaborn`, etc.
     
   * If you don't like messing up with your own environment, just run `make venv` in the terminal. This will create a virtual environment at `./venv` and install our specified dependencies. Our shell scripts (e.g., `./mai.sh`) will automatically activate and use this environment once it exists.
     
   * If you'd like our shell scripts to use and activate your own conda/virtualenv environment, feel free to edit the `.env.sh` under the environement activation section and add your custom activation lines. We source the `.env.sh` code in all of our shell scripts, so your changes will automatically have a global effect.
     
   </details>

</details>

<details open>
<summary><h2>Step-by-Step Guide to the Code</h2></summary>
   
+  <details>
   <summary><strong>Cloning the Repo</strong></summary>

   +  <details open>
      <summary><strong>[Option 1] Cloning All Three Repositories of Our Paper</strong></summary>
 
      1. `git clone --recursive https://github.com/ehsansaleh/firth_bias_reduction.git`
      2. `cd firth_bias_reduction/code_dcf`
      </details>
 
   +  <details>
      <summary><strong>[Option 2] Cloning This Repository Alone</strong></summary>
 
      1. `git clone https://github.com/sabagh1994/code_dcf.git`
      2. `cd code_dcf`
      </details>

   </details>
   
+  <details>
   <summary><strong>Download the Features</strong></summary>

   1. To use our pre-computed features, run `./features/download.sh`

   </details>
 
+  <details>
   <summary><strong>[Optional] Download the Cache Files</strong></summary>

   1. To use our pre-generated random-state cache files, run `./cache/download.sh`.
   2. These files essentially determine the few-shot task classes and their support and query sets.
   2. This downloading step can be skipped since the code can re-generate these files automatically.
   3. We only provided these files for better reproducibility and possibly faster initial run times.

   </details>
   
+  <details>
   <summary><strong>[Optional] Make a Virtual Environment</strong></summary>
   
   1. Activate your favorite python version (we used 3.8).
   2. Run `make venv`.
   3. This will take a few minutes, and about 1 GB in storage.
   4. The virtual environment with all dependencies will be installed at `./venv`.
   5. You can run `source ./venv/bin/activate` to activate the venv.
   6. Our shell scripts check for the existence of `venv`, and will use/activate it.
   
   </details>

+  <details>
   <summary><strong>Training Few-shot Classifiers</strong></summary>
   
   +  <details>
      <summary><strong>[Manual Approach]</strong></summary>
   
      * To fire up some training yourself, run

        `python main.py --device cuda:0 --configid "4_5ways/mini2mini_1s5w_750aug"`
      * This command will read the `./configs/4_5ways/mini2mini_1s5w_750aug.json` config as input.
      * The computed accuracy statistics would be saved at  `./results/4_5ways/mini2mini_1s5w_750aug.csv`.
      * Typically, this config may take 20 minutes to finish on a P100 or a V100 GPU.
      </details>
   
   +  <details open>
      <summary><strong>[Shell Script's Automated Array]</strong></summary>

      * Check-out and run [`./main.sh`](./main.sh).
      * The shell script performs some inital sanity checks and activations.
      * Then it will go through the `CFGPREFIXLIST` config array sequentially.
      * Feel free to add or take off configs from the array. 
      </details>
   
   </details>

+  <details>
   <summary><strong>Summarizing the Results</strong></summary>

   Run `make summary` 
   +  <details>
      <summary><strong>The Summary Output</strong></summary>

      This step generates the following 2 files.
      1. [`./summary/test.csv`](./summary/test.csv) summarizes the accuracy statistics on the novel split.
      2. [`./summary/test2test.csv`](./summary/test2test.csv) summarizes what happens when you apply the validated coefficients.
         * That is, what the accuracy improvements are when you pick the best coefficient from the validation set and apply it to the novel set.

      You can use these summarized CSV files to generate your own plots. Basically, `./summary/test2test.csv` has all the data we showed in our paper.
      </details>
   
   +  <details>
      <summary><strong>More Information</strong></summary

      Here are some pointers to understand what `make summary` just did:
      1. In the previous step, you have run a bunch of Few-shot classification tasks 
         1. on different datasets and augmentation settings,
         2. both when the firth bias reduction was turned on or off,
         3. etc.
      2. The statistics for each task were computed and stored in csv files in the results directory.
      3. Now, you wish to see how much difference Firth made after validation. 
         * This is what we call the summarization step.
      3. During the summarization
         1. we take all the generated `./results/*.csv` files from the previous step, and
         2. summarize them into a single small csv file at [`./summary/test2test.csv`](./summary/test2test.csv). 
      4. The [`./summary/test2test.csv`](./summary/test2test.csv) file includes
         1. the validated coefficients, 
         2. the average un-regularized accuracy values,
         3. the average accuracy improvement at test time, and
         4. what the error/confidence intervals look like

      as response columns. Each row will denote a specific configuration (e.g., dataset, number of shots, number of ways, etc. combination) averaged over many tasks.
       
      </details>

   </details>

+  <details>
   <summary><strong>Generating Our Tables</strong></summary>

   Run `make tables`. 
   
   * This will refresh the contents of the `tables` directory with new tex/csv tables.
   
   </details>

+  <details>
   <summary><strong>[Optional] Download The Pre-trained Feature Backbone Parameters</strong></summary>
   
   Run  `./checkpoints/download.sh`
  
    * These files were produced by the S2M2 project, and published at [their google drive](https://drive.google.com/drive/folders/1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_). The `./checkpoints/download.sh` only automates the downloading and placement process.
   
   </details>

+  <details>
   <summary><strong>[Optional] Downloading and Extracting the Datasets</strong></summary>
   
      Run `./Datasets/download.sh`
      
      1. Before you start, you should know that this can take a long time and a lot of storage.
  
         * For mini-imagenet, the download size is about 6.28 GBs, and the tar ball gets extracted to 60,000 files.
  
         * For CUB, the download size is about 1.06 GBs, and the tar ball gets extracted to 11,788 files.
  
         * For tiered-imagenet, the download size is about 82 GBs (divided into 6 download parts), and it ends up creating 779,165 files.
 
         * For CIFAR-FS, the download size is about 161 MBs, and the tar ball gets extracted to 60,000 files.
  
      2. This shell script will download and extract the mini-imagenet and CUB datasets by default.
         
      3. For tiered-imagenet, you can run `./Datasets/download.sh tiered`.
        
         * We suggest that you first do a plain `./Datasets/download.sh` run, since the other datasets are smaller to download and easier to check.
            
         * The tiered-imagnet dataset that we used is about 82GBs after compression into a single tar-ball. 
  
         * We divided this tar-ball into 6 parts, and the shell script will take care of stitching them together for extracting the images. 
  
         * If you want to save space after everything was extracted, you can manually remove these downloaded part files.
     
      4. For CIFAR-FS, you can run `./Datasets/download.sh cifar`.
      
      5. The script checks the existence and the MD5 hash of the downloaded files before downloading them. 
  
         * If the files already exist and are not damaged, the script will exit gracefully without downloading or extracting any files. 
         
   </details>

+  <details>
   <summary><strong>[Optional] Generating the Datasets filelists</strong></summary>
   
      Run `make filelists`
      
      1. You need to have the datasets downloaded and extracted before performing this step.
  
      2. One of the generated outputs is `./filelists/miniImagenet/base.json` for example.

         * The [`filelists/download.sh`](./filelists/download.sh) script downloads a set of template json filelists. 
           * The template json files include a list of image filenames and labels in the order we used them.
           * The template json files only include relative image paths, which should be converted to absolute paths using the `filelists/json_maker.py`](./filelists/json_maker.py) script.
  
         * The [`filelists/json_maker.py`](./filelists/json_maker.py) script generates these json files for all the `base`, `val`, and `novel` splits, and all the `miniImagenet`, `tieredImagenet`, `CUB` datasets by default.
  
         * You can specify your own list of splits and datasets at [`filelists/json_maker.py`](./filelists/json_maker.py) if you do not want all of the combinations to be generated. Look for and modify the `dataset_names` and `splits` variables to your liking in the python script.
  
         * The [`filelists/json_maker.py`](./filelists/json_maker.py) script makes random checks for the existence of the actual image files with a 1 percent chance.
     
      3. The feature generation scripts (e.g., `save_features.py`) use the generated `json` files as a reference for construcing datasets and data-loaders in pytorch. 
         
   </details>
 
+  <details>
   <summary><strong>[Optional] Train Feature Backbones</strong></summary>
    
    * You can use our [`code_s2m2rf`](https://github.com/ehsansaleh/firth_bias_reduction/tree/main/code_s2m2rf) project or the [original S2M2 project](https://github.com/nupurkmr9/S2M2_fewshot) to train new feature backbones.
    * Once you obtained new feature backbones, you can replace the trained checkpoints in the `checkpoints` directory or add new ones.
   
   </details>

   
+  <details>
   <summary><strong>[Optional] Generate Target Dataset Features using the Backbone Trained on the Source Dataset</strong></summary>
   
     * Here is a minimal python example:
       ```bash
       source .env.sh
       python save_features.py --source-dataset <source_dataset_name> \
                               --target-dataset <target_dataset_name> \
                               --split <split_name> --method S2M2_R \
                               --model WideResNet28_10
       ```
  
     * Our [`save_features.py`](./save_features.py) script is a modification of the [DC github repository's `save_plk.py` script](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration/blob/master/save_plk.py). 
 
     * By default, the pretrained backbones saved in the [`checkpoints`](./checkpoints) directory are used to generate the features.

     * The `split_name` can be chosen from `val`, `novel`, and `base`.
     
     * The `source_dataset_name` is the dataset on which the backbone is trained. It can be set to `miniImagenet`, `tieredImagenet`, and `CUB`. These are the datasets used for our paper's experiments.
 
     * The `target_dataset_name` is the dataset for which the features are extracted. It can be set to `miniImagenet`, `tieredImagenet`, and `CUB`.
 
     * Note that each time you run `save_features.py`, you will get a different ordering of the data points. 
       * This is because the `shuffle` argument for the `dataloader` is `True` in the original [script in DC github repository](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration/blob/5aab53eb4b5f102119ce9c71a6fda8b528cba48f/data/datamgr.py#L60).
       * We would have controlled this randomness by disabling the `shuffle` argument, but we noticed this issue only recently. 
       * Feel free to set `shuffle=True` if you want to disable this source of randomness. 
       * To reproduce the results of our paper, simply stick with the downloaded features since they have the same ordering we used in our paper.
  
   </details>

</details>
   
<details>
<summary><h2>Configurations and Arguments</h2></summary>

+ <details open>
  <summary><strong>Example</strong></summary>

  We have included all the configurations used to produce our papaer's results in the [`./configs`](./configs) directory. 

  There are a total of 16 json config files for the cross-domain experiments, and 8 config files for the tiered-imagenet dataset experiments.

  You can take a look at [`configs/1_mini2CUB/5s10w_0aug.json`](configs/1_mini2CUB/5s10w_0aug.json) for an example:
  ```json
  {
    "rng_seed": 0,
    "n_tasks": 10000,
    "source_dataset": "miniImagenet",
    "target_dataset": "CUB",
    "backbone_arch": "WideResNet28_10",
    "backbone_method": "S2M2_R",
    "n_shots_list": [5],
    "n_ways_list": [10],
    "split_list": ["novel"],
    "n_aug_list": [0],
    "firth_coeff_list": [0.0, 1.0],
    "n_query": 15,
    "dc_tukey_lambda": 0.5,
    "dc_k": 2,
    "dc_alpha": 0.21,
    "lbfgs_iters": 100,
    "store_results": true,
    "dump_period": 10000,
    "torch_threads": null
  }
  ```
  
  Our code runs the cartesian product of all arguments ending with `_list`. 
    * For instance, there is `2=1*1*1*1*2` different settings to try in the above config file.
    * Each of these settings runs 10,000 tasks, creating a total of 20,000 tasks to perform for this file.
 
  **Notes on Firth Coefficient Validation**: This project performed the validation of firth coefficient in a different manner than the other two [`code_firth`](https://github.com/ehsansaleh/firth_bias_reduction/tree/main/code_firth) and [`code_s2m2rf`](https://github.com/ehsansaleh/firth_bias_reduction/tree/main/code_s2m2rf) projcets.
     * Due to the time crunch at the review time, we did not run a full array of firth coefficients on both the validation and novel sets. 
     * Instead, we ran a small number of tasks for validation, identified the best firth coefficient in each setting, and then only ran the picked coefficient on the novel set with 10,000 tasks.
     * This approach reduced the computational cost by an order of magnitude.
     * The picked coefficients for each setting is specified in the config files.
     * For example, you can see in [`./configs/3_tiered2tiered/5s10w_0aug.json`](./configs/3_tiered2tiered/5s10w_0aug.json) that `"firth_coeff_list"` was set to `[0.0, 1.0]`. This means that the best firth coefficient was validated to be 1.
  </details>
  
+ <details>
  <summary><strong>Brief Argument Descriptions</strong></summary>
  
  * `"rng_seed"` determine the random seed to generate the set of 10,000 few-shot tasks.
  * `"n_tasks"` determines the number of few-shot tasks for evaluation of the approach.
  * `"source_dataset"` is the source dataset in cross-domain experiments.
    * This is the dataset from which the base classes for distribution calibration come from. 
    * That is, the `k` nearst neighbor classes for DC are chosen from the base split of the source dataset.
    * The features are extracted by a backbone network trained on the base split of the source dataset. 
    * The source dataset should be one of the `"miniImagenet"`, `"CUB"`, or `"tieredImagenet"` options.
  * `"targe_dataset"` is the targe dataset in cross-domain experiments.
    * This is the dataset from which the evaluation images and classes (novel or validation) are chosen.
    * The features used are extracted by the backbone trained on the base class of the source dataset. 
    * The target dataset should be one of the `"miniImagenet"`, `"CUB"`, or `"tieredImagenet"` options.
    * For _traditional non-cross-domain_ settings, you can set the source and target datasets to be the same. 
      * For example, all the json files under the [`configs/3_tiered2tiered`](./configs/3_tiered2tiered) and [`configs/4_5ways`](./configs/4_5ways) directories use the same target as the source dataset.
  * `"backbone_arch"` specifies the feature backbone architucture to use.
    * We only used the `WideResNet28_10` model in our experiments.
  * `"backbone_method"` specifies the feature backbone training algorithm to evaluate.
    * We only used feature backbones trained with the `S2M2_R` method in our experiments.
  * `"n_shots_list"` specifies a list of number of shots to test.
  * `"n_ways_list"` specifies a list of number of classes to perform few-shot classification tasks over.
  * `"split_list"` is a list of data splits to go through:
    * It should be a subset of `["base", "val", "novel"]`.
  * `"n_aug_list"` specifies a list of number of augmentation samples.
    * The augmented samples are sampled from the calibrated normal distribution.
    * The DC method suggests a default value of 750 for this step.
  * `"firth_coeff_list"` specifies a list of firth bias reduction coefficients to iterate over. 
  * `"n_query"` is the number of query samples to evaluate the accuracy the few-shot classifiers.
  * `"dc_tukey_lambda"` is the Tukey transformation parameter used to calibrate the normal distribution of features.
  * `"dc_k"` specifies the number of nearst neighbor base classes used to calibrate the normal distribution of features.
  * `"dc_alpha"` specifies the `alpha` parameter used calibrate the normal distributuion's convariance matrix.
  * `"lbfgs_iters"` specifies the number of L-BFGS iterations to train the few-shot classifier.
  * `"store_results"` should mostly be set to true, so that the python script writes its results in a `./results/*.csv` file.
    * If you just want to take dry-runs to check for code integrity, you can turn this option off.
  * `"torch_threads"` sets the number of torch threads.
    * This is just in case you wanted to train the classifiers on a CPU device. 
    * The code was optimized to require minimal CPU usage if a GPU was provided.
    * Therefore, you can safely set this to a small number when using a GPU.
    * You can set this option to `null` to keep the default value PyTorch sets.
  * `"dump_period"` specifies the number of CSV lines that need to be buffered before flushing them to the disk. 
    * This was set to a large value to prevent frequent disk dumps and causing system call over-heads.

  </details>

</details>

<details>
<summary><h2>Extra Inner-working Details</h2></summary>

+ <details>
  <summary><strong>Downloading the Files</strong></summary>

    You can find the google-drive download link embedded in the download shell-scripts. For example, take the following snippet from the [`./features/download.sh`](./features/download.sh) script:
     ```commandline
     FILEID="1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO"
     FILENAME="features.tar"
     GDRIVEURL="https://drive.google.com/file/d/1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO/view?usp=sharing"
     PTHMD5FILE="features.md5"
     REMOVETARAFTERDL="1"
     gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL}
     ```
     This means that you can manually
     1. download the file from [`https://drive.google.com/file/d/1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO/view?usp=sharing`](https://drive.google.com/file/d/1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO/view?usp=sharing),
     2. name it `features.tar`,
     3. optionally, verify its checksum from `features.md5`, and then
     4. untar it yourself, and you'll be in business!
     5. The `REMOVETARAFTERDL=1` option causes the script to remove the downloaded tar file upon completion.

    The function `gdluntar` from [`./utils/bashfuncs.sh`](./utils/bashfuncs.sh) is used to automatically download the files. We have been using this method for downloading google-drive files for quite a few years, and it's been stable so far. In the event there was a breaking change in google's api, please let us know and feel free to edit this function if you know a better G-drive download method in the meantime.
  
  </details>

+ <details>
  <summary><strong>Python Environments and Libraries</strong></summary>

  The [`.env.sh`](./.env.sh) checks for the existence of this virtual environment, and if it detects its existence, it will automatically activate and use it in our shell scripts. You can change this behavior by replacing the `[[ -f ${SCRIPTDIR}/venv/bin/activate ]] && source ${SCRIPTDIR}/venv/bin/activate` line with your own custom environment activation commands (such as `conda activate` or similar ones).

  </details>
  
</details>
   
## References
* Here is the arxiv link to our paper:
  * The arxiv PDF link: [https://arxiv.org/pdf/2110.02529.pdf](https://arxiv.org/pdf/2110.02529.pdf)
  * The arxiv web-page link: [https://arxiv.org/abs/2110.02529](https://arxiv.org/abs/2110.02529)
* Here is the open-review link to our paper:
  * The open-review PDF link: [https://openreview.net/pdf?id=DNRADop4ksB](https://openreview.net/pdf?id=DNRADop4ksB)
  * The open-review forum link: [https://openreview.net/forum?id=DNRADop4ksB](https://openreview.net/forum?id=DNRADop4ksB)
* Our paper got a spot-light presentation at ICLR 2022.
  * We will update here with links to the presentation video and the web-page on `iclr.cc`.
* Here is a web-page for our paper: [https://sabagh1994.github.io/firthfsl](https://sabagh1994.github.io/firthfsl)
* Here is the bibtex citation entry for our work:
```
@inproceedings{ghaffari2022fslfirth,
    title={On the Importance of Firth Bias Reduction in Few-Shot Classification},
    author={Saba Ghaffari and Ehsan Saleh and David Forsyth and Yu-Xiong Wang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=DNRADop4ksB}
}
```
