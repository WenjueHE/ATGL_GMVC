close all; clear all; clc
warning off
addpath('./data/MSRC');
addpath('./tools')
rng(2023)

currentFolder = pwd;
betas = [1e-1,1e1];
gammas = [1e-2,1e-1,1,1e1,1e2];
lambdas = 0;
ITERS = 2;
percentDels = 0.1;
is_missing = 1;
Dataname = 'MSRC';
load(Dataname);
num_cluster = length(unique(labels));
knns = [num_cluster,2*num_cluster,3*num_cluster,4*num_cluster,5*num_cluster,6*num_cluster,7*num_cluster,8*num_cluster,9*num_cluster];
num_views = length(data);
    
for i_perDel = 1:length(percentDels)
    X = data;
    percentDel = percentDels(i_perDel);
    Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
    load(Datafold);
    filename = strcat('ATGL-incomplete-',Dataname,'-',num2str(percentDel),'.txt');
    iter_folds = 1;
    ind_folds = folds{iter_folds};
    W = cell(1,num_views);
    Y = cell(num_views,1);
    X_exist = cell(num_views,1);
    for iv = 1:num_views
        X1 = X{iv};
        ind_0{iv} = find(ind_folds(:,iv) == 0);
        ind_1{iv} = find(ind_folds(:,iv) == 1);
        X1(ind_0{iv},:) = [];  
        Y{iv} = X1;         
    end
    clear X1 ind_0
    X = Y;
    clear Y

    for i_b = 1:length(betas)
        beta = betas(i_b);
        for i_l = 1:length(lambdas)
            lambda = lambdas(i_l);
            for i_g = 1:length(gammas)
                gamma = gammas(i_g);
                 res = zeros(ITERS,7);
                for i_knn = 1:length(knns)
                    knn = knns(i_knn);
                    for j =1:ITERS
                        tic;
                        [result,S,Tim] = ATGL_GMVC(X,labels,beta,knn,gamma,is_missing,ind_folds);%res = [acc, nmi, Purity, Fscore, Precision, Recall, ARI];
                        t = toc;
                        res(j,:)=result;
                        fprintf('beta = %d, lambda = %d\n',beta,lambda);
                        result
                        dlmwrite(filename,[beta,gamma,knn, result t],'-append','delimiter','\t','newline','pc');
                    end
                end
            end
        end
    end
end
