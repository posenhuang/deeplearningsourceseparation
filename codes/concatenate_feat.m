function feats =concatenate_feat(frame_feat, nprev, nnext)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% concatenate nprev + self + nnext frames together

    [nframes, ndim] = size(frame_feat);
    nDim= (nprev+nnext+1)*ndim;

    % duplicate the beginning and the end of frame
    begin_frame = repmat(frame_feat (1, :), nprev, 1);
    end_frame = repmat(frame_feat ( end, : ), nnext, 1) ;

    frame_feat2 = [begin_frame; frame_feat; end_frame];

    % original frame list
    ndx= nprev+1: nprev+nframes;

    % frame list + nprev + nnext frames
    ndx_merge= repmat( ndx, nprev+nnext+1, 1) + ...
               repmat( (0:nprev+nnext)'-nprev, 1, nframes) ;

    ndx_merge=ndx_merge(:);

    % merge them together
    feats=reshape( frame_feat2(ndx_merge,:)', nDim, nframes)';
end
