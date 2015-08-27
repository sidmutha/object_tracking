function [logical_cut,center]=Kernel_GraphCuts_Examples1(Imask,path,result,BW,iter1,center)
% addpath('D:\Program Files\MATLAB\R2010a\toolbox\stats\');

%This code implements multi-region graph cut image segmentation according
%to the kernel-mapping formulation in M. Ben Salah, A. Mitiche, and 
%I. Ben Ayed, Multiregion Image Segmentation by Parametric Kernel Graph
%Cuts, IEEE Transactions on Image Processing, 20(2): 545-557 (2011).
%The code uses Veksler, Boykov, Zabih and Kolmogorov’s implementation
%of the Graph Cut algorithm. Written in C++, the graph cut algorithm comes
%bundled with a MATLAB wrapper by Shai Bagon (Weizmann). The kernel mapping
%part was implemented in MATLAB by M. Ben Salah (University
%of Alberta). If you use this code, please cite the papers mentioned in the
%accompanying bib file (citations.bib).

%%%%%%%%%%%%%%%%%%%%%%%%%%%Requirements%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This code was tested with:
% • MATLAB Version: 7.12.0.635 (R2011a) for 32-bit wrapper
% • Microsoft Visual C++ 2010 Express

%%%%%%%%%%%%%%%%%%%Generating the mex files in MATLAB%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%>>mex -g GraphCutConstr.cpp graph.cpp GCoptimization.cpp Graph-
%Cut.cpp LinkedBlockList.cpp maxflow.cpp

%>>mex -g GraphCutMex.cpp graph.cpp GCoptimization.cpp GraphCut.cpp
%LinkedBlockList.cpp maxflow.cpp

% clear all; close all; 

%%%%%%%%%%%%%%%%%%%%%%%Main inputs and parameters%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Note: The RBF-kernel parameters are given in function kernel RBF.m%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%Example with a color image%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% path = 'Images\Brain_image.tif';
% im = im2double(imread(path)); 
% alpha=1; %The weight of the smoothness constraint
% k =8; %The number of regions

 


%%%%%%%Example with a SAR image corrupted with a multiplicative noise%%%%%%
%%%%%%%%%%%%%%%%Uncomment the following to run the example)%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% im=(double(I2));

% path = 'N:\research\DATA\STABILIZATION\frames_unstable\C\0051.jpg';%
% path = 'Images\frame0008.jpg';
im = im2double(imread(path));
alpha=0.5;
k =5;%size(center,1);

% Imask1 = Imask;
% [row,col] = find(Imask==1);
% Imask1(max(min(row)-30,1):min(max(row)+30,size(Imask,1)),max(min(col)-30,1):min(max(col)+30,size(Imask,2))) = 1;
% % im = zeros(size(im2));
% Imask_logical = Imask1==1;
% % imch1 = im2(:,:,1);
% % imch2 = im2(:,:,2);
% % imch3 = im2(:,:,3);
% % % imch = zeros(size(im2,1),size(im2,2));
% % imch = imch1(max(min(row)-30,1):size(Imask,1),max(min(col)-30,1):min(max(col)+30,size(Imask,2)));
% % im(:,:,1) = imch;
% % imch = imch2(max(min(row)-30,1):size(Imask,1),max(min(col)-30,1):min(max(col)+30,size(Imask,2)));
% % im(:,:,2) = imch;
% % imch = imch3(max(min(row)-30,1):size(Imask,1),max(min(col)-30,1):min(max(col)+30,size(Imask,2)));
% % im(:,:,3) = imch;

%%%%%%%%%%%%%%%%%%%%%%%%%%Example with a brain image%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%Uncomment the following to run the example)%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% path = 'Images\image0005.jpg';
% im = im2double(imread(path));
% load('Images\image.mat');
% % im = (im1);

% alpha=0.1;
% k =4;


sz = size(im);
Hc=ones(sz(1:2));
Vc=Hc;
i_ground = 0; % rank of the bakground for plotting, 0: the darkest; 
%k-1 the brightest; 99: nowhere

diff=10000;
an_energy=999999999;
iter=0;
iter_v=0;
energy_global_min=99999999;

distance = 'sqEuclidean'; % Feature space distance
im1(:,:,1:3) = im; 
% result1 = result(max(min(row)-30,1):min(max(row)+30,size(Imask,1)),max(min(col)-30,1):min(max(col)+30,size(Imask,2)),:);
im1(:,:,4) = 0.3*result;
% Initialization: cluster the data into k regions
tic,
disp('Start kmeans');
% im1=result;
data = ToVector(im1);
% min_data = min(data);
% max_data = max(data);

min_data = min(data,[],2);
max_data = max(data,[],2);
difference = max_data - min_data;
numerator = data - repmat(min_data,1,size(data,2));
data_normalized = numerator./repmat(difference,1,size(data,2));
% % if iter1~=1
% %     
% % [idx c] = kmeans(data, k, 'distance', distance,'EmptyAction','drop','maxiter',100,'start',center);
% % else
    [idx c] = kmeans(data, k, 'distance', distance,'EmptyAction','drop','maxiter',100);
    center = c;
% % end
c=c(find(isfinite(c(:,1))),:);                                                   
k=size(c,1);
k_max=k;
kmean_time=toc,

Dc = zeros([sz(1:2) k],'single');   
c,


tic
while iter < 5
    iter=iter+1;
    clear Dc
    clear K
    c;
    for ci=1:k
        K=kernel_RBF(im1,c(ci,:));
        Dc(:,:,ci)=1-K;
    end   
    clear Sc
    clear K
    %% The smoothness term
    Sc = alpha*(ones(k) - eye(k)); 
    gch = GraphCut('open', Dc, Sc, Vc, Hc);
    [gch L] = GraphCut('swap',gch);
    [gch se de] = GraphCut('energy', gch);
    nv_energy=se+de;
    gch = GraphCut('close', gch);
 
    if (nv_energy<=energy_global_min)
        diff=abs(energy_global_min-nv_energy)/energy_global_min;
        energy_global_min=nv_energy;
        L_global_min=L;
        k_max=k;
        
        nv_energy;
        iter_v=0;
        % Calculate region Pl of label l
        if size(im, 3)==3 % Color image
        for l=0:k-1
            Pl=find(L==l);
            card=length(Pl);
            K1=kernel_RBF(im1(Pl),c(l+1,1));
            K2=kernel_RBF(im1(Pl),c(l+1,2));K3=kernel_RBF(im1(Pl),c(l+1,3)); 
            K4=kernel_RBF(im1(Pl),c(l+1,4));%K5=kernel_RBF(im1(Pl),c(l+1,5)); K6=kernel_RBF(im1(Pl),c(l+1,6));
            smKI(1)=sum(im1(Pl).*K1); smKI(2)=sum(im(Pl+prod(sz(1:2))).*K2); smKI(3)=sum(im(Pl+2*prod(sz(1:2))).*K3);
            smKI(4)=sum(im1(Pl+3*prod(sz(1:2))).*K4);%smKI(5)=sum(im1(Pl+4*prod(sz(1:2))).*K5);smKI(6)=sum(im1(Pl+5*prod(sz(1:2))).*K6);
            smK1=sum(K1);smK2=sum(K2);smK3=sum(K3);
            smK4=sum(K4);%smK5=sum(K5);smK6=sum(K6);
            
            
         
            if (card~=0)
                c(l+1,1)=smKI(1)/smK1;c(l+1,2)=smKI(2)/smK2;c(l+1,3)=smKI(3)/smK3;
                c(l+1,4)=smKI(4)/smK4;
% c(l+1,5)=smKI(5)/smK5;c(l+1,6)=smKI(6)/smK6;
            else
                c(l+1,1)=999999999;c(l+1,2)=999999999;c(l+1,3)=999999999;
                c(l+1,4)=999999999;
%              c(l+1,5)=999999999;c(l+1,6)=999999999;
            end
        end
        end
        
        if size(im, 1)==1 % Gray-level image
        for l=0:k-1
            Pl=find(L==l);
            card=length(Pl);
            K=kernel_RBF(im(Pl),c(l+1,1));
            smKI=sum(im(Pl).*K);
            smK=sum(K);
            if (card~=0)
                c(l+1,1)=smKI/smK;
            else
                c(l+1,1)=999999999;
            end
        end
        end
        
        
        c=c(find(c(:,1)~=999999999),:);
        c_global_min=c;
        k_global=length(c(:,1));
        k=k_global;

    else
        iter_v=iter_v+1;
        %---------------------------------
        %       Begin updating labels
        %---------------------------------
        % Calculate region Pl of label l
        if size(im, 3)==3 % Color image 
        for l=0:k-1           
            Pl=find(L==l);
            card=length(Pl);
            K1=kernel_RBF(im1(Pl),c(l+1,1));K2=kernel_RBF(im1(Pl),c(l+1,2));K3=kernel_RBF(im1(Pl),c(l+1,3));
            K4=kernel_RBF(im1(Pl),c(l+1,4));%K5=kernel_RBF(im1(Pl),c(l+1,5));K6=kernel_RBF(im1(Pl),c(l+1,6));
            smKI(1)=sum(im1(Pl).*K1); smKI(2)=sum(im1(Pl+prod(sz(1:2))).*K2); smKI(3)=sum(im1(Pl+2*prod(sz(1:2))).*K3);
            smKI(4)=sum(im1(Pl+3*prod(sz(1:2))).*K4);%smKI(5)=sum(im1(Pl+4*prod(sz(1:2))).*K5);smKI(6)=sum(im1(Pl+5*prod(sz(1:2))).*K6);
            smK1=sum(K1);smK2=sum(K2);smK3=sum(K3);
            smK4=sum(K4);%smK5=sum(K5);smK6=sum(K6);
            % Calculate contour Cl of region Pl
            if (card~=0)
                c(l+1,1)=smKI(1)/smK1;c(l+1,2)=smKI(2)/smK2;c(l+1,3)=smKI(3)/smK3;
                c(l+1,4)=smKI(4)/smK4;%c(l+1,5)=smKI(5)/smK5;c(l+1,6)=smKI(6)/smK6;
            else
                c(l+1,1)=999999999;c(l+1,2)=999999999;c(l+1,3)=999999999;
                c(l+1,4)=999999999;%c(l+1,5)=999999999;c(l+1,6)=999999999;
                area(l+1)=999999999;
            end
        end
        end
        
        if size(im, 3)== 1 % Gray-level image 
        for l=0:k-1           
            Pl=find(L==l);
            card=length(Pl);
            K=kernel_RBF(im(Pl),c(l+1,1));
            smKI=sum(im(Pl).*K);
            smK=sum(K);
            % Calculate contour Cl of region Pl
            if (card~=0)
                c(l+1,1)=smKI/smK;
            else
                c(l+1,1)=999999999;
                area(l+1)=999999999;
            end
        end
        end
              
        c=c(find(c(:,1)~=999999999),:);
        k=length(c(:,1));
    end
end

L=L_global_min;
energy_global_min;
c=c_global_min;

size(c,1)
iter;

%Show the results

if size(im1, 3)>=3 % Color image 
img=zeros(sz(1),sz(2),3);
j=1;
imagesc(im); axis off; hold on; 

for i=0:k_max-1
    LL=(L_global_min==i);
    is_zero=sum(sum(LL));
    if is_zero
         img(:,:,1)=img(:,:,1)+LL*c(j,1);
         img(:,:,2)=img(:,:,2)+LL*c(j,2);
         img(:,:,3)=img(:,:,3)+LL*c(j,3);
         j=j+1;
    end
    if i~=i_ground
        color=[rand rand rand];
        contour(LL,[1 1],'LineWidth',2.5,'Color',color); hold on;
    end
end
figure(2);
imagesc(img); axis off;
end

if size(im1, 3)==1 % Gray-level image 
img=zeros(sz(1),sz(2));
j=1;
imagesc(im); axis off; hold on; colormap gray; 

for i=0:k_max-1
    LL=(L_global_min==i);
    is_zero=sum(sum(LL));
    if is_zero
         img(:,:,1)=img(:,:,1)+LL*c(j,1);
         j=j+1;
    end
    if i~=i_ground
        color=[rand rand rand];
        contour(LL,[1 1],'LineWidth',2.5,'Color',color); hold on;
    end
end
figure(2);
imagesc(img); axis off;
end
%% get cluster ids 
cluster_ids = unique(img(:,:,1));
%% logical of labelled data: BW
%% label and cluster ids logical : get the cluster ids of labelled data
count =0;
[row,col]=find(BW==1);
%% adjustment
BW_adjust = zeros(size(BW,1),size(BW,2));
min_col = min(col)+40;
max_col = max(col)-50;
% % min_col = round((min(col)+max(col))/2-35);
% % max_col = round((min(col)+max(col))/2+35);
min_row = min(row)+5;
max_row = max(row)-5;

BW_adjust(min_row:max_row,min_col:max_col) = BW(min_row:max_row,min_col:max_col);

for i =1:k
cluster = img(:,:,1)==cluster_ids(i) & BW_adjust==1;    
if nnz(cluster)~=0
    count = count+1;
    cluster_id_labelled(count,1) = cluster_ids(i);
   
end

end
%% logical of the mask
logical_mask = Imask==1;
%% mask and cluster id of labelled 

cutout = zeros(size(img,1),size(img,2),count);
temp = img(:,:,1);
for i =1:count
%% consider the connectedness of the pixels as well
cut = zeros(size(img,1),size(img,2));

    cut(logical_mask) = temp(logical_mask)==cluster_id_labelled(i);
    [rows,cols] = find(BW_adjust==1 & cut==1);
clear offset;
    offset(:,1) = rows;offset(:,2) = cols;
    %% change in cut
    cut(:,size(cut,2))=0;
    cut(size(cut,1),:)=0;
    
    cut_connected=get_connected_labelling(cut,offset);
cutout(:,:,i) = cut_connected;
end
logical_cut=sum(cutout,3)>=1;
hold on
h=imshow(BW_adjust);
set(h,'AlphaData',0.5);
hold off
pause(0.3);
end