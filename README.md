#Deep Learning For Monaural Source Separation

##Demo:
Webpage: https://sites.google.com/site/deeplearningsourceseparation/


###MIR-1K experiment (singing voice separation)

1. Training code: ```codes/mir1k/train_mir1k_demo.m```
 
2. Demo: 

    - Download a trained model 
    ```http://www.ifp.illinois.edu/~huang146/DNN_separation/model_400.mat```
    - Put the model at ```codes/mir1k/demo``` and go to the folder
    - Run: ```codes/mir1k/demo/run_test_single_model.m```


####TIMIT experiment (speech separation)
1. Training code: ```codes/timit/train_timit_demo.m```

2. Demo 
    - Download a trained model 
    ```http://www.ifp.illinois.edu/~huang146/DNN_separation/timit_model_70.mat```
    - Put the model at ```codes/timit/demo``` and go to the folder
    - Run: ```codes/timit/demo/run_test_single_model.m```


#### TSP experiment (speech separation)

1. Training code: ```codes/timit/train_timit_demo.m```

2. Demo
   - Download a trained model
    ```http://www.ifp.illinois.edu/~huang146/DNN_separation/TSP_model_RNN1_win1_h300_l2_r0_64ms_1000000_softabs_linearout_RELU_logmel_trn0_c1e-10_c0.001_bsz100000_miter10_bf50_c0_d0_7650.mat```
   - Put the model at ```codes/TSP/demo``` and go to the folder
   - Run the demo code at ```codes/TSP/demo/run_test_single_model.m```

#### Denosing experiment

1. Demo
    - Download a trained model
    ```http://www.ifp.illinois.edu/~huang146/DNN_separation/denoising_model_870.mat```
    - Put the model at ```codes/denoising/demo``` and go to the folder
    - Run the demo code at ```codes/denoising/demo/run_test_single_model.m```


###Dependencies
1. The package is modified based on rnn-speech-denoising.
Reference: https://github.com/amaas/rnn-speech-denoising

2. The software depends on Mark Schmidt's minFunc package for convex optimization.
Reference: http://www.di.ens.fr/~mschmidt/Software/minFunc.html

3. Additionally, we have included Mark Hasegawa-Johnson's HTK write and read functions
that are used to handle the MFCC files.
Reference: http://www.isle.illinois.edu/sst/software/

4. We use HTK for computing features (MFCC, logmel) (HCopy).
Reference: http://htk.eng.cam.ac.uk/

5. We use signal processing functions from labrosa.
Reference: http://labrosa.ee.columbia.edu/

6. We use BSS Eval toolbox Version 2.0, 3.0 for evaluation.
Reference: http://bass-db.gforge.inria.fr/bss_eval/

7. We use MIR-1K for singing voice separation task.
Reference: https://sites.google.com/site/unvoicedsoundseparation/mir-1k


###Work on your data:
1. To try the codes on your data, see mir1k setting - put your data into codes/mir1k/Wavfile accordingly.
 
2. Look at the unit test parameters below codes/mir1k/train_mir1k_demo.m. 

3. Tune the parameters and check the results.
 
###Reference
1. P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "[Joint Optimization of Masks and Deep Recurrent Neural Networks for Monaural Source Separation](http://posenhuang.github.io/papers/Joint_Optimization_of_Masks_and_Deep%20Recurrent_Neural_Networks_for_Monaural_Source_Separation_TASLP2015.pdf)", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 12, pp. 2136–2147, Dec. 2015

2. P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "[Singing-Voice Separation From Monaural Recordings Using Deep Recurrent Neural Networks](http://posenhuang.github.io/papers/DRNN_ISMIR2014.pdf)," in International Society for Music Information Retrieval Conference (ISMIR) 2014.

3. P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "[Deep Learning for Monaural Speech Separation](http://posenhuang.github.io/papers/DNN_Separation_ICASSP2014.pdf)," in IEEE International Conference on Acoustic, Speech and Signal Processing 2014.

