## Active Learning for IO-HMMs


### Dependencies and pacakges:

  - Code for IO-HMMs is built up on SSM (https://github.com/lindermanlab/ssm) toolbox; we added functions on top of the available repo.
  Installation of this package requires running the following on terminal: 
  ```
  pip install -e . 
  ```

  - Packages required to run scripts in this repo are available in al.yaml.

  - Note: For Gibbs sampling using polya-gamma augmentation, we used code from https://github.com/slinderman/pypolyagamma. It is part of al.yaml but can also be installed using:
  ```
  pip install pypolyagamma
  ```


### IOHMMs:

  - run_iohmm.py trains IO-HMMs using different input selection schemes (infomax_gibbs/ infomax_VI/ random) and stores all values (error/entropy/selected inputs) to plot in Results_IOHMM. This also provides different variants of Gibbs sampling for fitting the model: our laplace-based Gibbs (called 'gibbs'), Gibbs sampling with Polya-Gamma augmentation (gibbs_PG), and laplace-based Gibbs with parallel chains (called 'gibbs_parallel), which we compare in the supplement. Example: 
  ```
  python run_iohmms.py --seed 1 --input_selection infomax_gibbs --fitting_method gibbs
  ```
  
  - run_modelmismatch.py trains IO-HMMs using the inputs selected by a single GLM (the model-mismatch analysis) and stores all values to plot in Results_modelmismatch. It takes a value for random seed. Example: 
  ```
  python run_modelmismatch.py --seed 1
  ```
  
  - state_predictions.py performs the state inference analysis. 
  
  - plotting_nbs/plotIOHMM.ipynb plots results stored after running run_iohmm.py, and replicates Fig 5 from the paper. plotting_nbs/Suppfigs_IOHMM.ipynb replicates supplemental IO-HMM figures.

### MGLMs:
  - run_mglm.py trains MGLMs using active learning as well as random sampling and stores all values (error/entropy) to plot in Results_MGLM.
    It takes a value for random seed.
    Example: 
    ```
    python run_mglm.py --seed 1
    ```

  - plotting_nbs/plotMGLM.ipynb plots results stored after running run_mglm.py

  (Note that code for MGLM is independent of the SSM toolbox and only requries numpy/scipy/autograd) 

------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
