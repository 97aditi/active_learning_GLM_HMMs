# IO-HMMs

Code for ICML submission: Bayesian Active Learning for Discrete Latent Variable Models

Packages required:
  - Plotting requires:
     matplotlib
     notebook

  - Code for IO-HMMs is built up on SSM (https://github.com/lindermanlab/ssm) toolbox; we added functions on top of the available repo.
    Installation of this package requires running the following on terminal: 
      cd IO-HMMs/ssm
      pip install numpy cython
      cd ..
      pip install -e .

------------------------------------------------------------------------------------------------------------------------------------------------------------------
Contains scripts to generate figures (4,5) in the paper and the supplementary figs (S2, S3) The default setting of parameters in the script is to enable reproduction of our experiments. 

IOHMMs (Fig 3, 4, S2):
  - IOHMMs/run_iohmm.py trains IO-HMMs using different input selection schemes (infomax_gibbs/ infomax_VI/ random) and stores all values (error/entropy/selected inputs) to plot in Results_IOHMM. This also provides different variants of Gibbs sampling for fitting the model: our laplace-based Gibbs (called 'gibbs'), Gibbs sampling with Polya-Gamma augmentation (gibbs_PG), and laplace-based Gibbs with parallel chains (called 'gibbs_parallel), which we compare in the supplement.  
  Example: python run_iohmms.py --seed 1 --input_selection infomax_gibbs --fitting_method gibbs

  (For Gibbs sampling using polya-gamma augmentation, we used code from https://github.com/slinderman/pypolyagamma, available in this zip)

  - IOHMMs/run_modelmismatch.py trains IO-HMMs using the inputs selected by a single GLM (the model-mismatch analysis) and stores all values to plot in Results_modelmismatch.
  It takes a value for random seed. 
  Example: python run_modelmismatch.py --seed 1

  - IOHMMs/plotMLR.ipynb loads all stored values from Results_IOHMM, Results_modelmismatch and plots them to reproduce panels C and D.

  - IOHMMs/Suppfigs_IOHMM.ipynb loads all stored values from Results_IOHMM and plots them to reproduce Fig. S2.

  - state_predictions.py performs the state inference analysis and reproduces Fig. 5. 



MGLMs (Fig S3):
  - IOHMMs/run_mglm.py trains MGLMs using active learning as well as random sampling and stores all values (error/entropy) to plot in Results_MGLM.
    It takes a value for random seed.
    Example: python run_mglm.py --seed 1

  - IOHMMs/plotMGLM.ipynb loads all stored values from Results_MGLM and plots them to reproduce Fig 4.
  (Note that code for MGLM is independent of the SSM toolbox and only requries numpy/scipy/autograd) 

------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
