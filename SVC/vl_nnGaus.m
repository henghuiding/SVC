function out = vl_nnGaus(x, a, dzdy)
%VL_NNSIGMOID CNN sigmoid nonlinear unit.
%   Y = VL_NNSIGMOID(X) computes the sigmoid of the data X. X can
%   have an arbitrary size. The sigmoid is defined as follows:
%
%     SIGMOID(X) = 1 / (1 + EXP(-X)).
%
%   DZDX = VL_NNSIGMOID(X, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.

% Copyright (C) 2015 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

y = exp(-(a*x).^2);

if nargin <= 2 || isempty(dzdy)
  out = y ;
else
  out = dzdy .* (y .* (-2*a^2*x)) ;
end
