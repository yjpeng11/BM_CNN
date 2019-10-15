import cv2 as cv
import numpy as np
import os
import os.path
import shutil

folderdir1 = '/mnt/Data/4/ActionSimulation2/Human36skele/skeleall89/'
folderdir2 = 'input/'
folderdirout = 'reorganize/'
# os.mkdir(folderdir1+folderdirout)
os.makedirs(folderdir1 + folderdirout + 'jpeg/',exist_ok = True)
os.makedirs(folderdir1 + folderdirout + 'flow/u/',exist_ok = True)
os.makedirs(folderdir1 + folderdirout + 'flow/v/',exist_ok = True)
videolist = os.listdir(folderdir1 + folderdir2)
videolist.sort()

## read all videos in the dir
for x in range(0, len(videolist)):
    folname = videolist[x][:-4].replace(".", "_")
    print(folname)
    print(x+1)
    folderpath_image = folderdir1 + folderdirout + 'jpeg/' + folname
    folderpath_flow_u = folderdir1 + folderdirout + 'flow/u/' + folname
    folderpath_flow_v = folderdir1 + folderdirout + 'flow/v/' + folname
    if os.path.exists(folderpath_flow_v) and not len(os.listdir(folderpath_flow_v))== 0  and not len(os.listdir(folderpath_flow_u))== 0 and not len(os.listdir(folderpath_image))== 0:
        print('exist')
    else:
        cap = cv.VideoCapture(folderdir1 + folderdir2 + videolist[x])
        videolength = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        ret, frame1 = cap.read()
        frame_small = cv.resize(frame1,dsize=(500,500),interpolation = cv.INTER_CUBIC)
        os.makedirs(folderpath_image,exist_ok = True)
        name = "%06d" % 1
        cv.imwrite(folderpath_image +'/frame'+ name + '.jpg', frame_small)
        prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

        for frame in range(0, videolength):
            cap.set(1, frame)
            ret, frame2 = cap.read()
            name = "%06d" % (frame+1,)
            frame_small = cv.resize(frame2,dsize=(500,500),interpolation = cv.INTER_CUBIC)
            cv.imwrite(folderpath_image +'/frame'+ name + '.jpg', frame_small)
            next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

            flow = cv.calcOpticalFlowFarneback(prvs, next, flow=None,
                                              pyr_scale=0.5, levels=1, winsize=20, 
                                              iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            ximage = flow[...,0]
            yimage = flow[...,1]
            # resize
            ximage = cv.resize(ximage,dsize=(500,500),interpolation = cv.INTER_CUBIC)
            yimage = cv.resize(yimage,dsize=(500,500),interpolation = cv.INTER_CUBIC)
            if frame == 0:
                ximageall = ximage
                yimageall = yimage
            else:
                ximageall = np.dstack((ximageall, ximage))
                yimageall = np.dstack((yimageall, yimage))
            prvs = next
            
        ## normalize matrix
        maxx = ximageall.max()
        minx = ximageall.min()
        maxy = yimageall.max()
        miny = yimageall.min()
        # ximageallnorm = (ximageall - minx)*255/(maxx-minx)
        # yimageallnorm = (yimageall - miny)*255/(maxy-miny)
        ratio=10
        ximageallnorm = ximageall*ratio+127
        yimageallnorm = yimageall*ratio+127
        
        # make folder
        os.makedirs(folderpath_flow_u,exist_ok = True)
        os.makedirs(folderpath_flow_v,exist_ok = True)
        for i in range(0, videolength):
            name = "%06d" % (i+1,)
            cv.imwrite(folderpath_flow_u + '/frame'+ name + '.jpg', ximageallnorm[...,i])
            cv.imwrite(folderpath_flow_v + '/frame'+ name + '.jpg', yimageallnorm[...,i])
        cap.release()
