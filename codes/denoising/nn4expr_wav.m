function [s,r] = nn4expr_wav( nspkr, spkrt, nf, g, pst, lo, evl)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
%% Add wav component to training (mixture), test(mixture) in order to pass
%  information to drnn codes


% Evaluate all cases
if ~exist( 'evl', 'var')
	evl = 0;
end

% Index of training leave-out speaker, noise, gain and utterance
los = lo{1};
lon = lo{2};
lor = lo{3};
lou = lo{4};

% Setup 
warning( 'off', 'MATLAB:audiovideo:wavread:functionToBeRemoved')
warning( 'off', 'MATLAB:audiovideo:wavwrite:functionToBeRemoved')

%
% Load speech data
%

% Location of TIMIT files
if strcmp( computer, 'MACI64')
	p = 'Data/timit/';
else
	p = 'Data/timit/';
end

% Get TIMIT speaker directories
%[~,l] = system( ['find ' p ' -name "' spkrt '*" -type d']);  % linux, mac
%[~,l] = system( ['dir ' p ' /s | findstr /i ' spkrt '*']); % windows
% l = textscan( l, '%s'); l = l{1};
% if isempty( spkrt)
% 	l = l(2:end);
% end

l = {'Data/timit/FCJF0', 'Data/timit/FDAW0', 'Data/timit/FDML0', 'Data/timit/FECD0', ...
    'Data/timit/FETB0', 'Data/timit/FJSP0', 'Data/timit/FKFB0', 'Data/timit/FMEM0', ...
    'Data/timit/FSAH0', 'Data/timit/FSJK1', 'Data/timit/FSMA0', 'Data/timit/FTBR0', ...
    'Data/timit/FVFB0' 'Data/timit/FVMH0'};

% Decide how many speakers to keep
l = l(1:nspkr);

% Load the speech
so = cell( length( l), 1);
tx = cell( length( l), 1);
for i = 1:length( l)
	% Get the waveform
	d = dir( [l{i} '/*.WAV']);
	%so{i} = cellfun( @(x)wavread( [l{i} '/' x])', {d(1:end).name}, 'UniformOutput', 0);
    so{i} = cellfun( @(x)readsph( [l{i} '/' x])', {d(1:end).name}, 'UniformOutput', 0);
	so{i} = cellfun( @(x)x/std(x), so{i}, 'UniformOutput', 0);

	% And the transcription
	d = dir( [l{i} '/*.TXT']);
	tx{i} = cellfun( @(x)textread( [l{i} '/' x], '%s', 'delimiter', '\n'), {d(1:end).name}, 'UniformOutput', 0);
	for j = 1:length( tx{i})
		ii = find( tx{i}{j}{1} == ' ');
		tx{i}{j} = tx{i}{j}{1}(ii(2)+1:end-1);
	end
end
whos so


%
% Get noises
%

% Load all the noises to test with
no = cell( length( nf), 1);
for i = 1:length( nf)
	[no{i},sr] = wavread( nf{i});
	no{i} = resample( sum( no{i}, 2), 16000, sr)';
	no{i} = no{i} / std( no{i});
end
whos no


%
% Make mixtures
%

% Make all combinations of mixtures, speakers and gains
% keep track of the clean signals as well
n = cell( [length( so) length( no) length( g) 1]); % noise 
s = cell( [length( so) length( no) length( g) 1]); % source
m = cell( [length( so) length( no) length( g) 1]); % mixture t = cell( [length( so) length( no) length( g) 1]); % transcriptions
for i = 1:length( so)
	for j = 1:length( no)
		for l = 1:length( g)
			for k = 1:length( so{i})
				ns = no{j}(randi(end-length(so{i}{k})-1) + (1:length(so{i}{k})));
				n{i,j,l,k} = ns/std( ns);
				s{i,j,l,k} = so{i}{k};
				m{i,j,l,k} = s{i,j,l,k} + g(l) * n{i,j,l,k};
				t{i,j,l,k} = tx{i}{k};
			end
		end
	end
end
disp( size( m))


%
% Denoising procedure
%

% Decide if we should use the GPU
if gpuDeviceCount()
 	disp( 'Using GPU')
 	m2g = @(x)gpuArray( single( x));
else
	m2g = @(x)x;
end

% Prep search parameters
ps = paramcell( pst)

% Start looking
clear asr sar sir sdr paq psq sto r
tb = tic;
%parfor ip = 1:size( ps, 1)
for ip = 1:size( ps, 1)
	sz = ps{ip,1}.*[1 1/4];
	p  = ps{ip,2};
	em = ps{ip,3};
	it = ps{ip,4};
	hl = ps{ip,5};
	sg = ps{ip,6};
	nr = ps{ip,7};
%	fl = ps{ip,8};
	cf = ps{ip,9};    
	if size( ps, 2) > 9
		md = ps{ip,10};        
	else
% 		md = 'nn';
		md = 'drnn';        
    end
    
    if size( ps, 2) > 10
        circular_shift_size = ps{ip, 11};
    end    
    if size( ps, 2) > 11,
        max_lbfgs_iter = ps{ip,12};
    else
        max_lbfgs_iter = 500;
    end
    if size( ps, 2) > 12,
        feature_type = ps{ip,13};
    else
        feature_type = 'spectra';
    end
    if size( ps, 2) > 13,
        isRNN = ps{ip,14};
    else
        isRNN = 0;
    end
    
    % discriminative training gamma parameter
    if size( ps, 2) > 14,
        pos_neg_r = ps{ip,15};
    else
        pos_neg_r = 0;
    end
    
    % One output source or two
    if size( ps, 2) > 15,
        iscleanonly = ps{ip,16};
    else         
        iscleanonly = 0;
    end
    
      % batchsize
    if size( ps, 2) > 16,
        batchsize = ps{ip,17};
    else         
        batchsize = 0;
    end
        
    if size( ps, 2) > 17,
        lbfgs_iter = ps{ip,18};
    else         
        lbfgs_iter = 0;
    end        
     if size( ps, 2) > 18,
        grad_clip = ps{ip,19};
    else         
        grad_clip = 0;
     end       
     if size( ps, 2) > 19,
        wdecay = ps{ip,20};
    else         
        wdecay = 0;
    end        
    
    
    if strcmp(md, 'drnn') || strcmp(md, 'drnn_mini'), 
        % Training setting
        % context window size
        context_win = it;
        % hidden units
        hidden_units = hl(1); %1000;

        num_layers = numel(hl);
        % RNN temporal connection
%         isRNN = 0;
     
        % Circular shift step
        % circular_step = 100000;
        circular_step = circular_shift_size;
        % normalize input as L1 norm = 1
        isinputL1 = 0;
        % 0: MFCC, 1: logmel, 2: spectra
        switch feature_type,
            case 'logmel' 
                MFCCorlogMelorSpectrum = 1;
            case 'spectra'
                MFCCorlogMelorSpectrum = 2;
            otherwise
                error('Unknown feature_type');
        end
        % feature frame rate

        % framerate = 64;
        if sz(1) == 1024,
            framerate = 64;
        elseif sz(1) == 512,
            framerate = 32;
        else 
            error('Unknown framerate');
        end

        % discriminative training gamma parameter
        % pos_neg_r = 0.05;
%         pos_neg_r = 0;

        % 0: not using GPU, 1: using GPU
        if gpuDeviceCount()
            isGPU = 1;
        else 
            isGPU = 0;
        end

        % 0:'softlinear',1:'softabs', 2:'softquad', 3:'softabs_const',
        % 4:'softabs_kl_const'
        % opt = 1;    
        switch cf
            case 'euc'
                opt=1;
            case 'kl'
                opt=4;
            otherwise, 
                error('Unknown cost function');        
        end

        %% not changed
        isdropout = 0;

        % Last layer - linear or nonlinear
        outputnonlinear = 0;

        % 0: logistic, 1: tanh, 2: RELU
        act = 2; % let sg==2

        % constant for avoiding numerical problems
        const = 1e-10;
        % constant for avoiding numerical problems
        const2 = 0.001;

        train_mode = 0;

        % optimization iteration
        % max_iter = 500;
        max_iter = max_lbfgs_iter;
        
        clip = grad_clip;
    end
            
	% Go to freq
	wn = sqrt( hann( sz(1), 'periodic')); % hann window
	fm = cellfun( @(x)stft2( x, sz(1), sz(2), 0, wn), m, 'UniformOutput', 0); % freq mixture
	fs = cellfun( @(x)stft2( x, sz(1), sz(2), 0, wn), s, 'UniformOutput', 0); % freq source
	fn = cellfun( @(x)stft2( x, sz(1), sz(2), 0, wn), n, 'UniformOutput', 0); % freq noise

	% Add emphasis (for freq)
	W = linspace( 1, em, size( fm{1}, 1))';
	fm = cellfun( @(x)bsxfun( @times, x, W), fm, 'UniformOutput', 0);
	fs = cellfun( @(x)bsxfun( @times, x, W), fs, 'UniformOutput', 0);
	if strcmp( md, 'nmf')
		fn = cellfun( @(x)bsxfun( @times, x, W), fn, 'UniformOutput', 0);
	end

	% Static normalization
	if nr
		gi = cellfun( @(x)(sum( abs( x), 1)+eps), fm, 'UniformOutput', 0);
		fi = cellfun( @(x,y)bsxfun( @rdivide, 4*abs( x), y), fm, gi, 'UniformOutput', 0);
		go = cellfun( @(x)(sum( abs( x), 1)+eps), fs, 'UniformOutput', 0);
		fo = cellfun( @(x,y)bsxfun( @rdivide, 4*abs( x), y), fs, go, 'UniformOutput', 0);
	else
		fi = cellfun( @(x)abs( x)/(3*sz(1)), fm, 'UniformOutput', 0); % f input 
		fo = cellfun( @(x)abs( x)/(3*sz(1)), fs, 'UniformOutput', 0); % f output
		go = {};
	end

	% Play with contrast
	fi = cellfun( @(x)x.^p, fi, 'UniformOutput', 0);
	fo = cellfun( @(x)x.^p, fo, 'UniformOutput', 0);

	% Temporal stacking (t == 1 is no stacking)
	if it > 1
		for i = 1:numel( fi)
			tf = [];
			for j = 1:it
				tf = [tf; [zeros( size( fi{i},1), j-1) fi{i}(:,1:end-j+1)]];
			end
			fi{i} = tf;
		end
	end

	% Get indices of training (ri) and testing data (ti)
	[ri,ti] = traintestindex( size( m), los, lon, lor, lou); 
	if evl
		ti = 1:numel( m);
	end

	% Do we predict the gain?
	if ~isempty( go) || nr == 2
		go2 = [go{ri}];
		go2 = go2 / max( go2);
	else
		go2 = [];
	end

	% Learn the model
	switch md
        case 'drnn_mini'
			G = drnn( {m{ri}}, {s{ri}}, {n{ri}}, context_win, ...
                hidden_units, num_layers, isdropout, ...
                isRNN, iscleanonly, circular_step , isinputL1, ...
                MFCCorlogMelorSpectrum, framerate, pos_neg_r, outputnonlinear, ...
                opt, act, train_mode, const, const2, isGPU, max_iter, batchsize, lbfgs_iter, clip, wdecay);
        case 'drnn'
			G = drnn( {m{ri}}, {s{ri}}, {n{ri}}, context_win, ...
                hidden_units, num_layers, isdropout, ...
                isRNN, iscleanonly, circular_step , isinputL1, ...
                MFCCorlogMelorSpectrum, framerate, pos_neg_r, outputnonlinear, ...
                opt, act, train_mode, const, const2, isGPU, max_iter, [], [], clip, wdecay);
		case 'nn'
			G = nn7r3( m2g( [fi{ri}]), m2g( [[fo{ri}];go2]), ...
				hl, sg, [1000 1e-6], [.0001 1.01], 0, cf);
		case 'nmf'
			G = nmfsep( abs( [fs{ri}]), abs( [fn{ri}]), [50 30], 100);
		case 'sub'
			G = mean( abs( [fn{ri}]), 2);
		case 'regr'
			G = [[fo{ri}];go2] * pinv( [fi{ri}]);
		otherwise
			error( 'Unknown model')
	end

	% Test on the held-out data
	r{ip} = cell( size( m));
	sar{ip} = NaN*ones( size( m));
	sdr{ip} = sar{ip}; sir{ip} = sar{ip}; sto{ip} = sar{ip}; asr{ip} = sar{ip};
	for i = 1:length( ti)

		% Predict clean magnitude spectrogram
		switch md
            case 'drnn_mini' 
                f = double(drnn( m{ti(i)}, G)); 
            case 'drnn'
                f = double(drnn( m{ti(i)}, G));                
            case 'nn'
				f = double( nn7r3( fi{ti(i)}, G));
			case 'nmf'
				f = nmfsep( fi{ti(i)}, [], G, 100);
			case 'sub'
				f = max( bsxfun( @minus, fi{ti(i)}, .0005*G), 0);
			case 'regr'
				f = max( 0, net * fi{ti(i)});
		end

		% Deal with gain if needed
		if size( f, 1) > size( fm{ti(i)}, 1)
			gg = f(end,:); %filter( han(fl), 1, f(end,:));
%			  gg = medfilt1( gg, fl);
			f = bsxfun( @times, f(1:end-1,:), gg);
		elseif nr == 1
			f = bsxfun( @times, f, go{ti(i)});
		end
		f = f.^(1./p);

		% Get time sequence
        if strcmp(md, 'drnn') || strcmp(md, 'drnn_mini'),
            r{ip}{ti(i)} = stft2( f, sz(1), sz(2), 0, wn);
        else
            % fm{ti(i)}./abs( fm{ti(i)}) --> put the phrase back               
            r{ip}{ti(i)} = stft2( bsxfun( @rdivide, f, W) .* fm{ti(i)}./abs( fm{ti(i)}), sz(1), sz(2), 0, wn);
        end
        
%         save wav       
%         wavwrite(m{ti(i)}./(max(abs(m{ti(i)}))+1e-3), 16000,['model_demo/mixture',num2str(i),'.wav']);
%         wavwrite(r{ip}{ti(i)}./(max(abs(r{ip}{ti(i)}))+1e-3), 16000,['model_demo/separated_speech',num2str(i),'.wav']);
%         noise_temp = m{ti(i)}(1:min(size(m{ti(i)},2), size(r{ip}{ti(i)},2)))-r{ip}{ti(i)}(1:min(size(m{ti(i)},2), size(r{ip}{ti(i)},2)));
%         wavwrite(noise_temp./(max(abs(noise_temp))+1e-3), 16000,['model_demo/separated_noise',num2str(i),'.wav']);
%         wavwrite(s{ti(i)}./(max(abs(s{ti(i)}))+1e-3), 16000,['model_demo/original_speech',num2str(i),'.wav']);
%         wavwrite(n{ti(i)}./(max(abs(n{ti(i)}))+1e-3), 16000,['model_demo/original_noise',num2str(i),'.wav']);
             
		% Get separation stats
		[sdr{ip}(ti(i)),sir{ip}(ti(i)),sar{ip}(ti(i)),sto{ip}(ti(i))] = sep_perf( r{ip}{ti(i)}, [s{ti(i)};n{ti(i)}], 16000);

		% Get ASR stats
%		aa = casreval( r{ip}{ti(i)}, t{ti(i)});
%		asr{ip}(ti(i)) = aa{1}(1);
	end
	
end
sec2time( toc( tb))
disp( 'Done!')

clear s
s.sdr = sdr;
s.sir = sir;
s.sar = sar;
s.sto = sto;
