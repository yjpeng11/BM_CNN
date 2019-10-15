clear

for direction =1%:2
    if direction==1
        direct='left';
    else
        direct='right';
    end
    folder = 'localnomotion_8P_life1_inv';

    %%
    % saction = {'brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs',...
    %     'dive','draw_sword','dribble','drink','eat','fall_floor','fencing',...
    %     'flic_flac','golf','handstand','hit','hug','jump','kick_ball',...
    %     'kick','kiss','laugh','pick','pour','pullup','punch',...
    %     'push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball',...
    %     'shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault',...
    %     'stand','swing_baseball','sword_exercise','sword','talk','throw','turn',...
    %     'walk','wave'};
    %%
    % folderdir = '/home/cvllab/work/action7videos/fMRI_test0_theorytest/reorganize/rename/jpeg';
    folderdir = ['/mnt/Data/2/ActionCNN_simulation/ActionNN_Inversion/98action_opencv/' folder '/reorganize_' direct '/jpeg'];
    txtdir = ['/mnt/Data/2/ActionCNN_simulation/ActionNN_Inversion/98action_opencv/' folder ''];
    % txtdir = '/home/cvllab/work/action7videos/fMRI_test0_theorytest';
    % folderdir = '/home/cvllab/work/Two_stream_collection/Simulation_1/data/train1_HMDB51/input/jpeg';
    % txtdir = '/home/cvllab/work/temporal-segment-networks/data/CMU';
    cd (folderdir);
    allfolder = dir('*_*');
    % filename = 'test03_CMU_walking_PL_SpatialScram_right.txt';%'testlist01_Human36M_video.txt';
    filename = ['test01_' folder '_' direct '.txt'];
    
    fileID = fopen(fullfile(txtdir,filename),'a+');
    for i=1:size(allfolder,1)
        %     temp = strsplit(allfolder(i).name,'_');
        %     temp = temp{2};
        %     switch temp
        %         case 'Eating'
        %             category = 'eat';
        %         case 'Sitting'
        %             category = 'sit';
        %         case 'Smoking'
        %             category = 'smoke';
        %         case 'Walking'
        %             category = 'walk';
        %         case 'Biking'
        %             category = 'ride_bike';
        %         case 'Diving'
        %             category = 'dive';
        %         case 'Fencing'
        %             category = 'fencing';
        %         case 'GolfSwing'
        %             category = 'golf';
        %         case 'Punch'
        %             category = 'punch';
        %         case 'ride'
        %             category = 'ride_bike';
        %         otherwise
        %             category = temp;
        %     end
        category = 'walk';
        fprintf(fileID,'%s/%s.avi\n',category,allfolder(i).name);
    end
    fclose(fileID);
    
end
