close all; clear all; 
% clc
warning off
rng(2023)

currentFolder = pwd;
addpath('./data');
addpath('./tools');
betas = [1e-1,1e1];
gammas = [1e-2,1e-1,1,1e1,1e2];
lambdas = 0;
is_missing = 0;
dataname = 'BBCSport';
fprintf(dataname);
fprintf('\n');
load(dataname); 
data = fea';
labels = gt;

num_cluster = length(unique(labels));
num_view = length(data);
num_samp = length(labels);
knns = [num_cluster,2*num_cluster,3*num_cluster,4*num_cluster,5*num_cluster,6*num_cluster,7*num_cluster,8*num_cluster,9*num_cluster];
filename = ['ATGL-',dataname,'-complete','.txt'];
    
for i_b = 1:length(betas)
    beta = betas(i_b);
    for i_l = 1:length(lambdas)
        lambda = lambdas(i_l);
        for i_g = 1:length(gammas)
            gamma = gammas(i_g);
            for i_k = 1:length(knns)
                knn = knns(i_k);
                for j =1:1
                    fprintf('Running ATGL_GMVC ...\n')  %  Ours
                    tic;
                    [res,S,Tim] = ATGL_GMVC(data,labels,beta,knn,gamma,is_missing);%res = [acc, nmi, Purity, Fscore, Precision, Recall, ARI];
                    t = toc;
                    fprintf('result:\t%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n',[res t]);
                    dlmwrite(filename,[beta gamma knn res t],'-append','delimiter','\t','newline','pc');
                end
            end
        end
    end
end