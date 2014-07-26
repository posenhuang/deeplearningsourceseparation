%% cfg
CFGPath='C:\Users\huang146\Documents\MATLAB\Source_Separation_DNN\htk_features\';
addpath(CFGPath);

%% data
addpath('C:\Users\huang146\Documents\MATLAB\Source_Separation_DNN\Minje');

SavePath='C:\Users\huang146\Documents\MATLAB\Source_Separation_DNN\htk_features\';

%% 
[s1, fs]=wavread('female_train.wav');

b=2*s1;

filename='temp.wav';
wavwrite(b, fs, filename);
 
% scpname='temp.scp';
% fid= fopen(scpname,'w');
% fprintf(fid, '%s%s',,);
% fclose(fid);
%%
config='fbank_64ms_step32ms.cfg';
command=sprintf('HCopy -A -C %s%s %s %s%s', CFGPath,config,'temp.wav', SavePath, 'Out.fbank');
system(command)
[ DATA, HTKCode ] = htkread( 'Out.fbank' );

%% 
config='mfcc_64ms_step32ms.cfg';
command=sprintf('HCopy -A -C %s%s %s %s%s', CFGPath,config,'temp.wav', SavePath, 'Out.mfcc');
system(command)
[ DATA, HTKCode ] = htkread( 'Out.mfcc' );
