Deep Learning For Monaural Source Separation

https://sites.google.com/site/deeplearningsourceseparation/

Let me know if you have any question
Po-Sen Huang (huang146@illinois.edu)

Dependencies
====================
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


Getting Started
====================
MIR-1K experiment:

1. training: codes/mir1k/train_mir1k_demo.m
 
2. testing: codes/mir1k/run_test_single_mode.m

3. trained model: 
http://www.ifp.illinois.edu/~huang146/DNN_separation/model_400.mat
-> put the model at codes/mir1k/model_demo

TIMIT experiment:

1. codes/timit/train_timit_demo.m

(change baseDir to the path with this README file)


Your data:

1. To try the codes on your data, see mir1k setting - put your data into codes/mir1k/Wavfile accordingly.
 
2. Look at the unit test parameters below codes/mir1k/train_mir1k_demo.m. 

3. Tune the parameters and check the results.
 

TODO
====================
Add more unit tests, comments, timit example



Reference
====================
P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "Singing-Voice Separation From Monaural Recordings Using Deep Recurrent Neural Networks," in International Society for Music Information Retrieval Conference (ISMIR) 2014.

P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "Deep Learning for Monaural Speech Separation," in IEEE International Conference on Acoustic, Speech and Signal Processing 2014.
