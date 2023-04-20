% Multi-view Subspace Clustering on Topological Manifold
function [result, S, Tim] = ATGL_GMVC(data,labels, beta, knn,gamma,is_missing,folds)
% data: num_view * 1 cell,each view is a num_exist * d_v matrix (missing data already dropped)
% num_clus: number of clusters
% num_view: number of views
% k: number of adaptive neighbours
% labels: groundtruth of the data, num by 1

num_view = size(data,1);
num_samp = size(labels,1);
num_clus = length(unique(labels));
alpha = 1/num_view*ones(1,num_view);
NITER = 30;

% normalization
normData = 1;
if normData == 1
    for iv = 1:num_view
        data{iv} = NormalizeFea(data{iv},1);
    end
end

%  calculating A
A = cell(num_view,1);
for iv = 1:num_view
    A{iv} = constructW_PKN(data{iv}',knn);
end

% constructing G and W
G = cell(num_view,1);
W = cell(num_view,1);
B = cell(num_view,1); % B = GAG';
if is_missing == 1
    for iv = 1:num_view
        ind_0 = find(folds(:,iv) == 0);  % indexes of misssing instances
        G{iv} = diag(folds(:,iv));
        G{iv}(:,ind_0) = [];
        B{iv} = G{iv}*A{iv}*G{iv}';
        W{iv} = ones(num_samp,num_samp);
        W{iv}(:,ind_0) = 0;
        W{iv}(ind_0,:) = 0;
    end
else
    for iv = 1:num_view
        G{iv} = eye(num_samp);
        B{iv} = A{iv};
        W{iv} = ones(num_samp,num_samp);
    end
end

% initialization
Z = cell(num_view,1);
L = cell(num_view,1);
sumZ = zeros(num_samp);
C = cell(num_view,1);
Q = cell(num_view,1);
H = cell(num_view,1);
for v = 1:num_view
    Zv = G{v}*A{v}*G{v}';
    Dv = diag(sum(Zv)+eps);
    Lv = Dv - Zv;
    Lv = Dv^(-0.5)*Lv*Dv^(-0.5);
    Z{v} = Zv;
    L{v} = Lv;
    D{v} = Dv;
    sumZ = sumZ + Zv;
    clear Zv Lv
end 
 
% initialize S
S = sumZ/num_view;
I = eye(num_samp);

% initialize P (performing eigenvalue decomposition on sumZ) and Q
sumZ0 = sumZ - diag(diag(sumZ));
W_sumZ = (sumZ0+sumZ0')/2;
D_sumZ = diag(sum(W_sumZ));
L_sumZ = D_sumZ-W_sumZ;
[P_0,~,~] = eig1(L_sumZ,num_clus,0);
P = P_0(:,1:num_clus);
for iv = 1:num_view
    Q{iv} = Z{iv}'*P;
end
clear iv

% update ...
tic;
for iter = 1:NITER

    % constructing C^v：C^v_jk = \sum_i (S_ij/sqrt(D_jj)-S_ik/sqrt(D_kk))
    parfor iv = 1:num_view
        temp_Dv = sum(Z{iv})+eps;
        for ij = 1:num_samp
            for ik = 1:num_samp
                temp = 0;
                for ii = 1:num_samp
                    temp = temp + (S(ii,ij)/sqrt(temp_Dv(ij))-S(ii,ik)/sqrt(temp_Dv(ik)))^2;
                end
                C{iv}(ij,ik)=temp;
            end
        end
        H{iv} = P*Q{iv}';
    end
    clear iv ij ik ii

    % update Z_v
    for iv = 1:num_view
        T = (B{iv}.*W{iv}-alpha(iv)*0.25*C{iv}+gamma*H{iv})./(W{iv}+gamma);
        for num = 1:num_samp
            indnum = [1:num_samp];
            indnum(num) = [];
            Z{iv}(indnum',num) = (EProjSimplex_new(T(indnum',num)'))';
        end
        Dv = diag(sum(Z{iv})+eps);
        L{iv} = diag(sum(Z{iv})+eps) - Z{iv};
        L{iv} = Dv^(-0.5)*L{iv}*Dv^(-0.5);
    end
    clear iv

    % update P
    sumQZ = 0;
    for iv = 1:num_view
        sumQZ = sumQZ + Z{iv}*Q{iv};
    end
    clear iv
    [U,~,V]=svd(sumQZ,'econ');
    P = U*V';
    
    % update Q
    for iv = 1:num_view
        Q{iv}=Z{iv}'*P;
    end
    clear iv
    
    % update S
    iniS = S;
    S = zeros(num_samp);
    temp_A = beta*I;
    for v = 1:num_view
        temp_A = temp_A + alpha(v)* L{v};
    end
    clear v
    for ni = 1:num_samp
        index = find(iniS(ni,:)>0);
        b = 2*beta*I(ni,index);
        % solve z^T*A*z-z^T*b
        [si, ~] = fun_alm(temp_A(index,index),b);
        S(ni,index) = si';
    end
    S = (S+S')/2; 
    %
    % update w_v
    for v = 1:num_view
        alpha(v) = 1/(2*sqrt(trace(S'*L{v}*S)));
    end
    clear v
    
    
    % calculate objective value
    obj = beta*norm(S-I,'fro')^2;
    for iv = 1:num_view
        obj = obj + norm((Z{iv}-B{iv}).*W{iv},'fro')^2 + gamma*norm(Z{iv}-H{iv})+alpha(iv)*0.5*trace(S'*L{iv}*S);
    end
    OBJ(iter)=obj;
    if iter > 5 && abs(OBJ(iter)-OBJ(iter-1))/OBJ(iter) < 1e-3
        break
    end
end
fprintf('iter:%d\n',iter);
Tim = toc;
% result 
y = litekmeans(P,num_clus);
result = EvaluationMetrics(labels, y);
end


function [v, obj] = fun_alm(A,b)
if size(b,1) == 1
    b = b';
end

% initialize
rho = 1.5;
mu = 30;
n = size(A,1);
alpha = ones(n,1);
v = ones(n,1)/n;
% obj_old = v'*A*v-v'*b;

obj = [v'*A*v-v'*b];
iter = 0;
while iter < 10
    % update z
    z = v-A'*v/mu+alpha/mu;

    % update v
    c = A*z-b;
    d = alpha/mu-z;
    mm = d+c/mu;
    v = EProjSimplex_new(-mm);

    % update alpha and mu
    alpha = alpha+mu*(v-z);
    mu = rho*mu;
    iter = iter+1;
    obj = [obj;v'*A*v-v'*b];
end
end


function [x] = EProjSimplex_new(v, k)
%
% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%
if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);

else
    x = v0;
end
end
