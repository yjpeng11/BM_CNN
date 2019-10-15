clear
for direction =1%:2
    if direction==1
        direct='left';
    else
        direct='right';
    end
%     rng(1);
    
    folder = 'localnomotion_8P_life1_inv';
    % folderstr= 'CMU/SpatialScram_PL';
    folderstr = ['98action_opencv/' folder];
    %%
    % saction = {'brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs',...
    %       'dive','draw_sword','dribble','drink','eat','fall_floor','fencing',...
    %       'flic_flac','golf','handstand','hit','hug','jump','kick_ball',...
    %       'kick','kiss','laugh','pick','pour','pullup','punch',...
    %       'push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball',...
    %       'shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault',...
    %       'stand','swing_baseball','sword_exercise','sword','talk','throw','turn',...
    %       'walk','wave'};
%    saction = {'Directions','Discussion','Eating','Greeting','Phoning','Posing','Purchases','Sitting',...
%    'SittingDown','Smoking','TakingPhoto','Waiting','Walking','WalkingDog','WalkTogether'};
    saction = {'eat','sit','smoke','walk'};
    %% get hmdb51 all video's dir within category
    % cd '/home/cvllab/work/HMDB-51/hmdb51_org';
    % cd (['/home/cvllab/work/temporal-segment-networks/data/' folderstr]);
    cd (['/mnt/Data/2/ActionCNN_simulation/ActionNN_Inversion/' folderstr]);
    fileID = fopen(['test01_' folder '_' direct '.txt']); %'testlist01_Human36M_video.txt','UCF101_5temp1.txt'
    a=textscan(fileID,'%s');
    a=a{1};
    fclose(fileID);
    
    allset = struct('cat',[],'act',[]);
    
    for ii = 1:length(saction)
        allset(ii).cat = saction{ii};
        allset(ii).act = {};
    end
    
    for ii = 1:length(a)
        temp = strsplit(a{ii},'/');
        allset(find(strcmp(saction,temp{1}))).act = [allset(find(strcmp(saction,temp{1}))).act; temp{2}];
    end
    
    
    
    % cate = dir('*');
    % cate = cate(3:end);
    
    % allset = struct('cat',[],'act',[]);
    %
    % for ii = 1:length(cate)
    %     allset(ii).cat = cate(ii).name;
    %     cd (cate(ii).name);
    %     action = [];
    %     action = dir('*.avi');
    %     howmanyact = length(action);
    %     allset(ii).act = {action.name};
    %     cd ..
    % end
    
    %%  get the img and flow dirs
    % jpegdir = ['/home/cvllab/work/temporal-segment-networks/data/' folderstr '/reorganize_right/rename/jpeg'];
    % tvl1dir = ['/home/cvllab/work/temporal-segment-networks/data/' folderstr '/reorganize_right/rename/flow'];
    jpegdir = ['/mnt/Data/2/ActionCNN_simulation/ActionNN_Inversion/' folderstr '/reorganize_' direct '/jpeg'];
    tvl1dir = ['/mnt/Data/2/ActionCNN_simulation/ActionNN_Inversion/' folderstr '/reorganize_' direct '/flow'];
    
    %% select test and train sets
    % portion = 0.9; % 80% train 20% test
    % count_train = 1;
    count_test = 1;
    % trainx = struct('videoname',[],'imgdir',[],'flowdiru',[],'flowdirv',[]);
    testx = struct('videoname',[],'imgdir',[],'flowdiru',[],'flowdirv',[]);
    % trainy = struct('category',[],'catnum',[]);
    testy = struct('category',[],'catnum',[]);
    
    for ii = 1:length(allset)
        numact = length(allset(ii).act);
        %     temp = randperm(numact);
        %     numtrain = round(numact*portion);
        for jj = 1:numact
            %         if jj<=numtrain
            %             trainx(count_train).videoname = allset(ii).act{temp(jj)};
            %             trainx(count_train).imgdir = fullfile(jpegdir,allset(ii).act{temp(jj)}(1:end-4));
            %             trainx(count_train).flowdiru = fullfile(tvl1dir,'u',allset(ii).act{temp(jj)}(1:end-4));
            %             trainx(count_train).flowdirv = fullfile(tvl1dir,'v',allset(ii).act{temp(jj)}(1:end-4));
            %             trainy(count_train).category = allset(ii).cat;
            %             trainy(count_train).catnum = find(not(cellfun('isempty', strfind(saction,allset(ii).cat))));
            %
            %             count_train = count_train+1;
            %         else
            testx(count_test).videoname = allset(ii).act{(jj)};
            testx(count_test).imgdir = fullfile(jpegdir,allset(ii).act{(jj)}(1:end-4));
            testx(count_test).flowdiru = fullfile(tvl1dir,'u',allset(ii).act{(jj)}(1:end-4));
            testx(count_test).flowdirv = fullfile(tvl1dir,'v',allset(ii).act{(jj)}(1:end-4));
            testy(count_test).category = allset(ii).cat;
            testy(count_test).catnum = find(not(cellfun('isempty', strfind(saction,allset(ii).cat))));
            count_test = count_test+1;
            %         end
        end
    end
    
    trainx=testx;
    trainy=testy;
    % cd (['/home/cvllab/work/temporal-segment-networks/data/' folderstr])
    cd (['/mnt/Data/2/ActionCNN_simulation/ActionNN_Inversion/' folderstr])
    
    save(['test01_' folder '_' direct '.mat'],'allset','saction','trainx','trainy','testx','testy');
    
    %% split the file
% %     clear
%     load(['test01_' folder '_' direct '.mat']);
%     splitpoint = 6000;
%     testx = testx(1:splitpoint);
%     testy = testy(1:splitpoint);
%     trainx=testx;
%     trainy=testy;
%     save(['test01_' folder '_' direct '_split1.mat'],'allset','saction','trainx','trainy','testx','testy');
%     
% %     clear
%     splitpoint = 6000;
%     load(['test01_' folder '_' direct '.mat']);
%     testx = testx(splitpoint+1:end);
%     testy = testy(splitpoint+1:end);
%     trainx=testx;
%     trainy=testy;
%     save(['test01_' folder '_' direct '_split2.mat'],'allset','saction','trainx','trainy','testx','testy');
end
