import os
import cPickle
import scipy.io as sio
import numpy as np


def label2vec(label, num_classes):
    """convert label into a one-hot vector"""
    vec = np.zeros((1, num_classes))
    vec[0, label] = 1
    return vec


def get_file_list(folder_dir):
    """get file list in a folder"""
    return [os.path.join(folder_dir, f) for f in os.listdir(folder_dir) \
            if os.path.isfile(os.path.join(folder_dir, f))] 


def prepare_data(X_data, Y_data, num_classes, name2id):
    """read, reconstruct & return data"""
    if len(X_data) != len(Y_data):
        print "Error: different number of instances in X and Y!"
        return None
    X_image = []
    X_flow_u = []
    X_flow_v = []
    video_lengths = []
    Y = []
    video_ids = []
    cur_label_id = 0
    #for video_id, x, y in enumerate(zip(X_data, Y_data)):
    for x, y in zip(X_data, Y_data):
        x_image_prefix = x[1][0]
        x_flow_u_prefix = x[2][0]
        x_flow_v_prefix = x[3][0]
        x_image_path = get_file_list(x_image_prefix)
        x_image_path.sort()
        #print x_image_path
        x_flow_u_path = get_file_list(x_flow_u_prefix)
        x_flow_v_path = get_file_list(x_flow_v_prefix)
        x_flow_u_path.sort()
        x_flow_v_path.sort()
        length = min(len(x_image_path), len(x_flow_u_path))
        cnt = length / 10
        for t in xrange(0, cnt * 10, 10):
            X_image.append(x_image_path[t:(t+10)])
            #print "----------------------"
            #print x_image_path[t:(t+10)]
            X_flow_u.append(x_flow_u_path[t:(t+10)])
            X_flow_v.append(x_flow_v_path[t:(t+10)])
        video_lengths.append(cnt)		
        label_name = y[0][0]
        if label_name not in name2id:
            print "error: can not find label", label_name 
            return None
        label = name2id[label_name] - 1
		# Y += [label2vec(label, num_classes)] * length
        Y += ([label] * cnt)
        #video_ids += ([video_id] * cnt)
        #print "-----------------------------------------------------"

    print len(X_image), len(X_flow_u), len(X_flow_v), len(video_lengths), sum(video_lengths), len(Y)

    return {'image': X_image,
            'flow_u': X_flow_u,
            'flow_v': X_flow_v, 
            'lengths': video_lengths, 
		    'labels': Y}
            #'video_ids': video_ids}


NUM_CLASSES = 2   # 15 51 4
#name2id = {'brush_hair': 1, 'cartwheel':2,'catch':3, 'chew':4,'clap':5,'climb':6,'climb_stairs':7, 'dive': 8,'draw_sword':9,'dribble':10, 'drink':11,'eat':12, 'fall_floor':13,'fencing': 14,'flic_flac':15, 'golf': 16, 'handstand':17,'hit':18,'hug':19,'jump':20,'kick_ball':21,'kick':22,'kiss':23,'laugh':24,'pick':25,'pour':26,'pullup':27,'punch': 28, 'push':29,'pushup':30,'ride_bike':31,'ride_horse':32,'run':33,'shake_hands':34,'shoot_ball':35,'shoot_bow':36,'shoot_gun':37,'sit':38, 'situp':39,'smile':40,'smoke':41,'somersault':42,'stand':43,'swing_baseball':44,'sword_exercise':45,'sword':46,'talk':47,'throw':48,'turn':49,'walk':50,'wave':51} # for HMDB-51
#name2id = {'eat': 1,'sit': 2, 'smoke': 3, 'walk': 4}
#name2id = {'direction': 1, 'discuss':2,'eat':3, 'greet':4,'phone':5,'pose':6,'purchase':7, 'sit': 8,'sitdown':9,'smoke':10, 'photo':11,'wait':12, 'walk':13,'walkdog': 14,'walktogether':15}
#name2id = {'left':1,'right':2}  # for left right softmax
name2id = {'causal':1,'moonwalk':2}  # for left right softmax
#name2id = {'eat': 3,'sit': 38, 'smoke': 41, 'walk': 50} # for Human3.6
#name2id = {'dive': 8, 'fencing': 14, 'golf': 16, 'punch': 28, 'ride_bike': 31} # for UCF 101
#name2id = {'dive': 8, 'fencing': 14, 'golf': 16, 'punch': 28, 'ride_bike': 31} # for HMDB51

#data_path = "/mnt/Data/2/ActionCNN_simulation/ActionNN_Inversion/98action_opencv/localnomotion_8P_life1_inv/test01_localnomotion_8P_life1_inv_left.mat"
data_path = "/mnt/Data/4/ActionSimulation2/Moonwalk/causal/causal_moonwalk/test01_causal_moonwalk_leftright_0.8.mat"

data = sio.loadmat(data_path)
train_data = prepare_data(data['trainx'][0], data['trainy'][0], NUM_CLASSES, name2id)
cPickle.dump(train_data, open("./hmdb51/flow_train_data.pik", "wb"), protocol = 2)
test_data = prepare_data(data['testx'][0], data['testy'][0], NUM_CLASSES, name2id)
cPickle.dump(test_data, open("./hmdb51/flow_test_data.pik", "wb"), protocol = 2)


# file_list = [f for f in os.listdir(path_prefix) if os.path.isfile.isfile(os.path.join(path_prefix, f))]
