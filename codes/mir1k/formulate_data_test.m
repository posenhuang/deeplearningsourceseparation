function  varargout = formulate_data_test(dmix, eI, mode)
% M: Number of sample files to use. value < 0 loads all
% eI.winSize: Size of window
% eI.seqLen: unique lengths (in ascending order)
%             files are chopped by these lengths. (ex: [1, 10, 100])
% eI.targetWhiten: Specify the path to the whitening data.
% data_ag: noisy data. cell array of different training lengths.
% target_ag: clean data. cell array of different training lengths.

% mode:
%     0: Training (noisy data and clean data)
%     1: Testing (just noisy data)
%     2: Error testing (noisy data and clean data, both loaded without chunking)

unique_lengths = [];

%% Set up. During testing, dont know the lengths so cant pre-allocate
if mode,
    data_ag = {};
    target_ag = {};  % returns empty targets
    mixture_ag = {};
    item=0;
else,
    seqLenSizes = zeros(1,length(eI.seqLen));
    % input feature calculate
    [DATA, mixture_spectrum, eI]=compute_features(dmix, eI);
    [T, nfeat] = size(DATA');

    remainder = T;
    for i=length(eI.seqLen):-1:1
      num = floor(remainder/eI.seqLen(i));
      remainder = mod(remainder,eI.seqLen(i));
      seqLenSizes(i) = seqLenSizes(i)+num;
    end
    data_ag = cell(1,length(eI.seqLen));
    target_ag = cell(1,length(eI.seqLen));
    mixture_ag = cell(1,length(eI.seqLen));

  for i=length(eI.seqLen):-1:1
    data_ag{i} = zeros(eI.inputDim*eI.seqLen(i),seqLenSizes(i));
    target_ag{i} = zeros(2*nfeat*eI.seqLen(i),seqLenSizes(i));
    mixture_ag{i} = zeros(nfeat*eI.seqLen(i),seqLenSizes(i));
  end
end

seqLenPositions = ones(1,length(eI.seqLen));

[DATA, mixture_spectrum, eI]=compute_features(dmix, eI);

multi_data = DATA;
[nFeat,T] = size(multi_data);

%% input normalize
if eI.inputL1==1, % DATA (NUMCOFS x nSamp)
%         apply CMVN to input
    cur_mean = mean(multi_data, 2);
    cur_std = std(multi_data, 0, 2);
    multi_data = bsxfun(@minus, multi_data, cur_mean);
    multi_data = bsxfun(@rdivide, multi_data, cur_std);
elseif eI.inputL1==2,
    l1norm = sum(multi_data,1)+eps;
    multi_data = bsxfun(@rdivide, multi_data, l1norm);
end

%% zero pad
if eI.num_contextwin > 1
    % winSize must be odd for padding to work
    if mod(eI.num_contextwin,2) ~= 1
        fprintf(1,'error! winSize must be odd!');
        return
    end;
    % pad with repeated frames on both sides so im2col data
    % aligns with output data
    nP = (eI.num_contextwin-1)/2;
    multi_data = [repmat(multi_data(:,1),1,nP), multi_data, ...
        repmat(multi_data(:,end),1,nP)];
end

%% im2col puts winSize frames in each column
multi_data_slid = im2col(multi_data,[nFeat, eI.num_contextwin],'sliding');
% concatenate noise estimate to each input
 if mode == 1, % Testing
    c = find(unique_lengths == T);
    if isempty(c)
        % add new unique length if necessary
        data_ag = [data_ag, multi_data_slid(:)];
        mixture_ag=[mixture_ag, mixture_spectrum(:)];

        unique_lengths = [unique_lengths, T];
    else
        data_ag{c} = [data_ag{c}, multi_data_slid(:)];
        mixture_ag{c} = [mixture_ag{c}, mixture_spectrum(:)];
    end;
elseif mode == 2, % Error analysis.
	c = find(unique_lengths == T);
    if isempty(c)
        % add new unique length if necessary
        data_ag = [data_ag, multi_data_slid(:)];
        unique_lengths = [unique_lengths, T];
    else
        data_ag{c} = [data_ag{c}, multi_data_slid(:)];
    end;
 elseif mode == 3 % formulate one data per cell
% 		c = find(unique_lengths == T);
    item =item+1;
    % feadim x nframes
    data_ag{item} = multi_data_slid;
    mixture_ag{item} = mixture_spectrum;

 else % training
	%% put it in the correct cell area.
	while T > 0
		% assumes length in ascending order.
		% Finds longest length shorter than utterance
		c = find(eI.seqLen <= T, 1,'last');

		binLen = eI.seqLen(c);
		assert(~isempty(c),'could not find length bin for %d',T);
		% copy data for this chunk
		data_ag{c}(:,seqLenPositions(c))=reshape(multi_data_slid(:,1:binLen),[],1);
        mixture_ag{c}(:,seqLenPositions(c))=reshape(mixture_spectrum(:,1:binLen),[],1);

        seqLenPositions(c) = seqLenPositions(c)+1;
		% trim for next iteration
		T = T-binLen;
		if T > 0
			multi_data_slid = multi_data_slid(:,(binLen+1):end);
            mixture_spectrum = mixture_spectrum(:,(binLen+1):end);
		end;
	end;
end;

theoutputs = {data_ag, target_ag, mixture_ag};
varargout = theoutputs(1:nargout);

return;

%% Unit test
% (TODO) add
eI.MFCCorlogMelorSpectrum=2; % 0- mfcc, 1- logmel, 2- spectrum
eI.winsize = 1024;    eI.nFFT = 1024;    eI.hop =eI.winsize/2;    eI.scf=1;
eI.featDim =513;
eI.num_contextwin=3;
eI.inputDim = eI.featDim * eI.num_contextwin;

[train1, fs, nbits]=wavread('female_train.wav');
[train2, fs, nbits]=wavread('male_train.wav');

maxLength=max([length(train1), length(train2)]);
train1(end+1:maxLength)=eps;
train2(end+1:maxLength)=eps;

train1=train1./sqrt(sum(train1.^2));
train2=train2./sqrt(sum(train2.^2));

eI.seqLen = [1 50 100];
eI.inputL1=0;eI.outputL1=0;
eI.fs=fs;
formulate_data_test(train1+train2, eI, 3)
end

