function train_denoising_demo()
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
%% Global parameters
rand('state',0)
randn('state',0)

addpath('drnn');
addpath(['Data',filesep,'noise']);
pg.nspkr = 10;
pg.spkrt = 'F';
pg.nf = {'babble.wav', 'drill.wav', 'airport.wav', 'train.wav', 'subway.wav'};
pg.g = 1;

% Parameter search space
pst = {
    {1024} % FFT size
    {1} % Power to raise spectra
    {1} % Emphasis
    {1} % Temporal stacking
    {[1000,1000]} % Hidden layers; 1000 optimal?
    {'poslin'} % Activations, poslin only is best
    {0} % Normalization flag
    {3} % Gain filtering (3 is no filtering)
    {'euc'} % Cost function
%         {'drnn'}
    {'drnn_mini'}
    {10000} % circular shift size 10000
    {20} % max_outer_iter
    {'spectra'}
    {1, 3} % isRNN
    {0}  % discriminative training gamma parameter
    {0}  % isclean only
    {100} %batchsize,
    {100} % lbfgs_iter 100
    {0} % gradient clipping
    {10} % weight norm
%         {5, 20, 30, 40} % weight norm
}; 

% Index of training leave-out speaker, noise, gain and utterance
los = [];
lon = [];
lor = [];
lou = [1 2];

[s,~] = nn4expr_wav( pg.nspkr, pg.spkrt, pg.nf, pg.g, pst, {los,lon,lor,lou});
save DRNN_wdecay10 s pst

load DRNN_wdecay10 s pst
figure(3)

[p,m,n] = nanstats2( s, [4 2 1]);
bare3( 1:4, p', m', n', .8, {{}}, 'Effect of Network Topology')
set( gcf, 'name', 'Layers', 'paperposition', [0.25 5.75 8 2.75])
legend({'RNN1', 'RNN3'}, 'fontsize', 8, 'location', 'northwest')
