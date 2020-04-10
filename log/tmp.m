% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc

ver = {'reslstm-0c', 'reslstm-0d', 'reslstm-0e'};

for i = 1:length(ver)
    T = readtable([ver{i}, '.csv']);
    epoch = 1:height(T);
    plot(epoch, T.loss); xlabel('Epoch'); ylabel('Training loss'); hold on;
end
legend(ver);
hold off;