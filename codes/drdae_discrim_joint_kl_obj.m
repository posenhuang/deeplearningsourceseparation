function [ cost, grad, numTotal, pred_cell ] = drdae_discrim_joint_kl_obj( ...
    theta, eI, data_cell, targets_cell, mixture_spectrum, fprop_only, pred_out)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% discrim. training + joint masking
%
%PRNN_OBJ MinFunc style objective for Deep Recurrent Denoising Autoencoder
%   theta is the full parameter vector
%   eI contains experiment / network architecture
%   data_cell is a cell array of matrices. Each a distinct length is a cell
%             entry. Each matrix has a time series example in each column
%   targets_cell is parallel to data, but contains the labels for each time
%   fprop_only is a flag that only computes the cost, no gradient
%   numTotal is total number of frames evaluated
%   pred_out is a binary flag for whether pred_cell is populated
%            pred_cell only filled properly when utterances one per cell


%% Debug: Turns this into an identity-function for debugging rest of system
if isfield(eI, 'objReturnsIdentity') && eI.objReturnsIdentity
    cost = 0; grad = 0; numTotal = 0;
    for l = 1:numel(data_cell)
        numUtterances = size(data_cell{l}, 2);
        original_vector = reshape(data_cell{l}, eI.winSize*eI.featDim, []);
        midPnt = ceil(eI.winSize/2);
        original_vector = original_vector((midPnt-1)*14+1 : midPnt*14, :);
        pred_cell{l} = reshape(original_vector, [], numUtterances);
    end
    return;
end

%if isempty(return_activation),
  return_activation = 0;
%end

%% Load data from globals if not passed in (happens when run on RPC slave)
global g_data_cell;
global g_targets_cell;
isSlave = false;
if isempty(data_cell)
    data_cell = g_data_cell;
    targets_cell = g_targets_cell;
    isSlave = true;
end;
pred_cell = cell(1,numel(data_cell));
act_cell = cell(1,numel(data_cell));
%% default short circuits to false
if ~isfield(eI, 'shortCircuit')
    eI.shortCircuit = 0;
end;

%% default dropout to false
if ~isfield(eI, 'dropout')
  eI.dropout = 0;
end;

%% setup weights and accumulators
[stack, W_t] = rnn_params2stack(theta, eI);
cost = 0; numTotal = 0;
outputDim = eI.layerSizes(end);

%% setup structures to aggregate gradients
stackGrad = cell(1,numel(eI.layerSizes));

if isfield(eI, 'fullRNN') && eI.fullRNN==1
   W_t_grad = cell(1,numel(eI.layerSizes)-1);
    for l = 1:numel(eI.layerSizes)-1
        W_t_grad{l}.W = zeros(size(W_t{l}.W));
    end
else
   W_t_grad = zeros(size(W_t));
end


for l = 1:numel(eI.layerSizes)
    stackGrad{l}.W = zeros(size(stack{l}.W));
    stackGrad{l}.b = zeros(size(stack{l}.b));
end
if eI.shortCircuit
    stackGrad{end}.W_ss = zeros(size(stack{end}.W_ss));
end;
%% check options
if ~exist('fprop_only','var')
    fprop_only = false;
end;
if ~exist('pred_out','var')
    pred_out = false;
end;

% DROPOUT: vector of length of hidden layers with 0 or 1
% (to drop or keep activation unit) with prob=0.5
hActToDrop = cell(numel(eI.layerSizes),1);
for i=1:numel(eI.layerSizes)-1
 if eI.dropout
   hActToDrop{i} = 1/eI.dropout * binornd(1,eI.dropout, eI.layerSizes(i),1);
   %hActToDrop{i} = round(rand(eI.layerSizes(i),1));
 else
   hActToDrop{i} = ones(eI.layerSizes(i),1);
 end
end

%% loop over each distinct length
for c = 1:numel(data_cell)
    if isempty(data_cell{c}), continue; end

    data = data_cell{c};
    targets = {};
    if ~isempty(targets_cell), targets = targets_cell{c}; end;
    uttPred = [];
    T =size(data,1) / eI.inputDim;
    % store hidden unit activations at each time instant
    hAct = cell(numel(eI.layerSizes)-1, T);
    for t = 1:T
        %% forward prop all hidden layers
        for l = 1:numel(eI.layerSizes)-1
            if l == 1
                hAct{1,t} = stack{1}.W * data((t-1)*eI.inputDim+1:t*eI.inputDim, :);
            else
                hAct{l,t} = stack{l}.W * hAct{l-1,t};
            end;
            hAct{l,t} = bsxfun(@plus, hAct{l,t}, stack{l}.b);
            % temporal recurrence. limited to single layer for now
            if t > 1
                if isfield(eI, 'fullRNN') && eI.fullRNN==1
                    hAct{l,t} = hAct{l,t} + W_t{l}.W * hAct{l,t-1};
                elseif l == eI.temporalLayer
                    hAct{l,t} = hAct{l,t} + W_t * hAct{l,t-1};
                end
            end;

            % nonlinearity
            if strcmpi(eI.activationFn,'tanh')
                hAct{l,t} = tanh(hAct{l,t});
            elseif strcmpi(eI.activationFn,'logistic')
                hAct{l,t} = 1./(1+exp(-hAct{l,t}));
            elseif strcmpi(eI.activationFn,'RELU')
                hAct{l,t} = max(0,hAct{l,t});
            else
                error('unrecognized activation function: %s',eI.activationFn);
            end;
            %dropout (hActToDrop will be all ones if no dropout specified)
            hAct{1,t} = bsxfun(@times, hAct{1,t}, hActToDrop{l});
        end;
        % forward prop top layer not done here to avoid caching it
    end;
    %% compute cost and backprop through time
    if  eI.temporalLayer
        if isfield(eI, 'fullRNN') && eI.fullRNN==1
            delta_t = cell(1, numel(eI.layerSizes)-1);
            for l = 1:numel(eI.layerSizes)-1
                delta_t{l} = zeros(eI.layerSizes(l),size(data,2));
            end
        else
            delta_t = zeros(eI.layerSizes(eI.temporalLayer),size(data,2));
        end
    end;

    y1_dim= 1:outputDim/2;
    y2_dim= outputDim/2+1:outputDim;
    mixtures=mixture_spectrum{c};

    for t = T:-1:1
        l = numel(eI.layerSizes);
        %% forward prop output layer for this timestep
        curPred = bsxfun(@plus, stack{l}.W * hAct{l-1,t}, stack{l}.b);

        if eI.outputnonlinear==1,
           if strcmpi(eI.activationFn,'tanh')
                curPred = tanh(curPred);
           elseif strcmpi(eI.activationFn,'logistic')
                curPred = 1./(1+exp(-curPred));
           elseif strcmpi(eI.activationFn,'RELU')
                curPred = max(0,curPred);
           else
                error('unrecognized activation function: %s',eI.activationFn);
           end
        end

        mixture=mixtures((t-1)*numel(y1_dim)+1:t*numel(y1_dim),:);
        a1 = curPred(y1_dim,:); a2 = curPred(y2_dim,:);

        const=eI.const;%1e-8 ;
        const2=eI.const2;% 1e-3;

        if strcmp(eI.opt,'softlinear'),
            y1= (a1)./((a1)+(a2)+1e-10).* mixture;
            y2= (a2)./((a1)+(a2)+1e-10).* mixture;
        elseif strcmp(eI.opt,'softabs'),
            y1= abs(a1)./(abs(a1)+abs(a2)+1e-10).* mixture;
            y2= abs(a2)./(abs(a1)+abs(a2)+1e-10).* mixture;
        elseif strcmp(eI.opt,'softabs_const') || strcmp(eI.opt,'softabs_kl_const'),
            y1= abs(a1)./(abs(a1)+abs(a2)+const).* mixture;
            y2= abs(a2)./(abs(a1)+abs(a2)+const).* mixture;
        elseif strcmp(eI.opt, 'softquad')
            y1= (a1.^2)./((a1.^2)+(a2.^2)+1e-10).* mixture;
            y2= (a2.^2)./((a1.^2)+(a2.^2)+1e-10).* mixture;
        else
        end
        weighted_curPred=[y1; y2];

        % add short circuit to regression prediction if model has it
        if eI.shortCircuit
            weighted_curPred = weighted_curPred + stack{end}.W_ss ...
                * data((t-1)*eI.inputDim+1:t*eI.inputDim, :);
        end;
        if pred_out, uttPred = [weighted_curPred, uttPred]; end;
        % skip loss computation if no targets given
        if isempty(targets), continue; end;

        curTargets = targets((t-1)*outputDim+1:t*outputDim, :);
        curTargets_neg = [curTargets(outputDim/2+1:outputDim,:); curTargets(1:outputDim/2,:)];

        y_t = (1- eI.r) * weighted_curPred + eI.r * curTargets_neg - curTargets;

        ya_ta= y_t(y1_dim,:);
        yb_tb= y_t(y2_dim,:);

        if strcmp(eI.opt,'softlinear'),
            delta_y1 =  (ya_ta-yb_tb).* y2./(a1+a2+1e-10);
            delta_y2 = (-ya_ta+yb_tb) .* y1./ (a1+a2+1e-10);
        elseif strcmp(eI.opt,'softabs'),
            delta_y1 =  (ya_ta-yb_tb).* y2./(abs(a1)+abs(a2)+1e-10);
            delta_y2 = (-ya_ta+yb_tb) .* y1./ (abs(a1)+abs(a2)+1e-10);

            delta_y1(a1<0) = -delta_y1(a1<0);
            delta_y2(a2<0) = -delta_y2(a2<0);
        elseif strcmp(eI.opt,'softabs_const'),
             const_div=const./((abs(a1)+abs(a2)+const).^2).* mixture;
             delta_y1 =  (ya_ta-yb_tb).* y2./(abs(a1)+abs(a2)+const);
             delta_y1 = delta_y1+ ya_ta.* const_div;
             delta_y2 = (-ya_ta+yb_tb) .* y1./ (abs(a1)+abs(a2)+const);
             delta_y2= delta_y2+ yb_tb.* const_div;

             delta_y1(a1<0) =  -delta_y1(a1<0);
             delta_y2(a2<0) =  -delta_y2(a2<0);
        elseif strcmp(eI.opt,'softabs_kl_const'),
             y_target_a = curTargets(y1_dim, :);
             y_target_b = curTargets(y2_dim, :);
             y_target_neg_a = y_target_b;
             y_target_neg_b = y_target_a;

             y_pred_a = weighted_curPred(y1_dim, :);
             y_pred_b = weighted_curPred(y2_dim, :);

             const_div=const./((abs(a1)+abs(a2)+const).^2).* mixture;

             delta_y1 =  (-y_target_a./(y_pred_a+const2) + y_target_b./(y_pred_b+const2)).* y2./(abs(a1)+abs(a2)+const);
             delta_y1 = delta_y1+  (-y_target_a./(y_pred_a+const2)+1).* const_div;

             delta_y2 =  (y_target_a./(y_pred_a+const2) - y_target_b./(y_pred_b+const2)).* y1./(abs(a1)+abs(a2)+const);
             delta_y2= delta_y2+ (-y_target_b./(y_pred_b+const2)+1) .* const_div;

             % discrim part
             delta_y1 =  delta_y1- eI.r* (-y_target_neg_a./(y_pred_a+const2) + y_target_neg_b./(y_pred_b+const2)).* y2./(abs(a1)+abs(a2)+const);
             delta_y1 = delta_y1-   eI.r* (-y_target_neg_a./(y_pred_a+const2)+1).* const_div;

             delta_y2 = delta_y2- eI.r*(y_target_neg_a./(y_pred_a+const2) - y_target_neg_b./(y_pred_b+const2)).* y1./(abs(a1)+abs(a2)+const);
             delta_y2= delta_y2-  eI.r*(-y_target_neg_b./(y_pred_b+const2)+1) .* const_div;

             delta_y1(a1<0) =  -delta_y1(a1<0);
             delta_y2(a2<0) =  -delta_y2(a2<0);
        elseif strcmp(eI.opt, 'softquad')
            delta_y1 =  (ya_ta-yb_tb).* (2*a1.*y2)./ (a1.^2+a2.^2+1e-10); %  y1= (a1.^2)./((a1.^2)+(a2.^2)+1e-8);%.* mixture;
            delta_y2 =  (-ya_ta+yb_tb).* (2*a2.*y1)./ (a1.^2+a2.^2+1e-10);%         y2= (a2.^2)./((a1.^2)+(a2.^2)+1e-8);%.* mixture;
        else
        end

        delta = [ delta_y1; delta_y2 ];
        if strcmp(eI.opt,'softlinear') || strcmp(eI.opt,'softabs') || strcmp(eI.opt, 'softquad') || strcmp(eI.opt,'softabs_const'),
             cost = cost + 0.5 * ( sum( sum((weighted_curPred - curTargets).^2)) ...
                -  eI.r* sum( sum((weighted_curPred - curTargets_neg).^2)));
        elseif strcmp(eI.opt,'softabs_kl_const'),
             cost = cost +...
             sum(sum( curTargets.*log( curTargets./(weighted_curPred + const2) + const2 )-curTargets+ weighted_curPred +const2))...
            -eI.r* sum(sum( curTargets_neg.*log( curTargets_neg./(weighted_curPred + const2) + const2 )-curTargets_neg+ weighted_curPred +const2));
        else

        end

        if eI.outputnonlinear==1,
             if strcmpi(eI.activationFn,'tanh')
                delta = delta .* (1 -curPred.^2);
            elseif strcmpi(eI.activationFn,'logistic')
                delta = delta .* curPred .* (1 - curPred);
            elseif strcmpi(eI.activationFn,'RELU')
                delta = delta .* double(curPred>0);
            else
                error('unrecognized activation function: %s',eI.activationFn);
            end;
        end

        if fprop_only, continue; end;
        %% regression layer gradient and delta
        stackGrad{l}.W = stackGrad{l}.W + delta * hAct{l-1,t}';
        stackGrad{l}.b = stackGrad{l}.b + sum(delta,2);
        % short circuit layer

        if eI.shortCircuit
            stackGrad{end}.W_ss = stackGrad{end}.W_ss + delta ...
                * data((t-1)*eI.inputDim+1:t*eI.inputDim, :)';
        end;
        delta = stack{l}.W' * delta;
        %% backprop through hidden layers
        for l = numel(eI.layerSizes)-1:-1:1
            % aggregate temporal delta term if this is the recurrent layer
            if isfield(eI, 'fullRNN') && eI.fullRNN==1
                delta = delta + delta_t{l};
            elseif l == eI.temporalLayer
              delta = delta + delta_t;
            else
            end
            % push delta through activation function for this layer
            % tanh unit choice assumed
            if strcmpi(eI.activationFn,'tanh')
                delta = delta .* (1 - hAct{l,t}.^2);
            elseif strcmpi(eI.activationFn,'logistic')
                delta = delta .* hAct{l,t} .* (1 - hAct{l,t});
            elseif strcmpi(eI.activationFn,'RELU')
                delta = delta .* double(hAct{l,t}>0);
            else
                error('unrecognized activation function: %s',eI.activationFn);
            end;
            % gradient of bottom-up connection for this layer
            if l > 1
                stackGrad{l}.W = stackGrad{l}.W + delta * hAct{l-1,t}';
            else
                stackGrad{l}.W = stackGrad{l}.W + delta * data((t-1)*eI.inputDim+1:t*eI.inputDim, :)';
            end;
            % gradient for bias
            stackGrad{l}.b = stackGrad{l}.b + sum(delta,2);

            % compute derivative and delta for temporal connections
            if t > 1
                if isfield(eI, 'fullRNN') && eI.fullRNN==1
                     W_t_grad{l}.W = W_t_grad{l}.W + delta * hAct{l,t-1}';
                     % push delta through temporal weights
                     delta_t{l} = W_t{l}.W' * delta;
                elseif l == eI.temporalLayer
                     W_t_grad = W_t_grad + delta * hAct{l,t-1}';
                     % push delta through temporal weights
                     delta_t = W_t' * delta;
                end
            end
            % push delta through bottom-up weights
            if l > 1
                delta = stack{l}.W' * delta;
            end;
        end
        % reduces avg memory usage but doesn't reduce peak
        %hAct(:,t) = [];
    end
    pred_cell{c} = uttPred;
    % Return the activations for this utterance.
    if return_activation,
      act_cell{c} = cell2mat(hAct);
    end
    % keep track of how many examples seen in total
    numTotal = numTotal + T * size(targets,2);
end


%% stack gradients into single vector and compute weight cost
wCost = numTotal * eI.lambda * sum(theta.^2);
grad = rnn_stack2params(stackGrad, eI, W_t_grad, true);
grad = grad + 2 * numTotal * eI.lambda * theta;

%% clipping
if isfield(eI,'clip') && eI.clip~=0, % if eI.clip==0, no clip
  if eI.clip > 0 % method one -clip the whole
      norm_grad = norm(grad);  
      fprintf('norm_grad:%f\n', norm_grad);
      % avoid numerial problem
      if norm_grad <0 || norm_grad > 1e15 || isnan(norm_grad) || isinf(norm_grad),
          grad = zeros(size(grad));
          fprintf('set gradient to zeros\n');
      end  
      if norm_grad > eI.clip 
         grad = eI.clip * grad/ norm_grad;    
      end  
  else % method two - clip each entry
      clip_value = -1*eI.clip;
      grad(grad > clip_value)=clip_value;
      grad(grad < -clip_value)=-clip_value;     
  end
end

%%
avCost = cost/numTotal;
avWCost = wCost/numTotal;
cost = cost + wCost;

% print output
if ~isSlave && ~isempty(targets_cell)
    fprintf('loss:  %f  wCost:  %f \t',avCost, avWCost);

    if isfield(eI, 'fullRNN') && eI.fullRNN==1
        fprintf('wNorm: %f  rNorm: %f  oNorm: %f\n',sum(stack{1}.W(:).^2),...
            sum(W_t{1}.W(:).^2), sum(stack{end}.W(:).^2));
    else
        fprintf('wNorm: %f  rNorm: %f  oNorm: %f\n',sum(stack{1}.W(:).^2),...
            sum(W_t(:).^2), sum(stack{end}.W(:).^2));
    end
end;
