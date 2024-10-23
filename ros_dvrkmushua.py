#!/usr/bin/env python
#computing kinematics of objects...
#Author:Gbenimachor Alex
#Date: 29/3/2019
#phil 4:13 (I can do all things through Christ which strengtheneth me.)
#import necessary libraries required of computation...
import cv2
import sys
import vrep
import time
import math
import rospy
import roslib
import tkinter
import numpy as np
import transforms3d
import message_filters
from  tkinter import *
from numpy.linalg import inv
from nav_msgs.msg import Odometry
from scipy.linalg import expm, logm
from PIL import Image as img, ImageFilter
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from numpy.linalg import inv, multi_dot, norm

#######################################################
PI = math.pi
cs = math.cos
sn = math.sin
#atan2 =math.atan2(y, x)
######################################################

#first lets get the joint handle...
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
if clientID != -1:
   print ('Connected to remote API server')
else: 
    print("not connected to remote API server")
    sys.exit("could not connect to V-REP")

rospy.init_node("vrep_catkin_ws", anonymous=True)
rospy.loginfo("Alex is just started the Simulation")
#sets the status message
returnCode=vrep.simxAddStatusbarMessage(clientID,"Simulation Started by Alex",vrep.simx_opmode_streaming)

#define the variable...
global T_2in0, deg, rad #Goal pose invthetaspsm2
global J1, J2, J3 #joint base...
global theta, M, S, invthetaspsm2 #theta..
global robvelocity, VelJ6, VelJG #velocity...
global Mu_indJ1, Mu_indJ2, Mu_indJ3   #Joints index
global Mu_midJ1, Mu_midJ2, Mu_midJ3  #Joints middle
global Mu_thuJ1, Mu_thuJ2, Mu_thuJ3  #Joints thumb

#variable of PSM1, manipulator...
global J1_PSMone, J2_PSMone, J3_PSMone #joint base...
global Mu_indJ1_PSMone, Mu_indJ2_PSMone, Mu_indJ3_PSMone   #Joints index
global Mu_midJ1_PSMone, Mu_midJ2_PSMone, Mu_midJ3_PSMone  #Joints middle
global Mu_thuJ1_PSMone, Mu_thuJ2_PSMone, Mu_thuJ3_PSMone  #Joints thumb





#LETS DEFINE SOME NEEDED FUNCTION
def inv_bracket(m):
    """
    Performs the inverse 'bracket' operation on a 3x3 or 4x4 matrix
    :param m: the 3x3 skew-symmetric matrix or 4x4 bracket of a twist - Must be convertible to a numpy array!
    :returns: the vector or twist representation of the input matrix or an empty list otherwise
    """
    rtn = []
    m = np.asarray(m)
    if(m.shape == (4,4)):
        rtn = np.block([[ inv_bracket(m[:3,:3])],
                        [ m[:3,3:]             ]])
    elif(m.shape == (3,3)):
        m = m - m.transpose()
        rtn = np.zeros((3,1))
        rtn[2] = - m[0][1]/2
        rtn[1] =   m[0][2]/2
        rtn[0] = - m[1][2]/2
    return rtn

 
def bracket(v):
    """
    Returns the 'bracket' operator of a 3x1 vector or 6x1 twist
    :param v: the 3x1 vector or 6x1 twist, can be of type list or numpy.ndarray - Must be convertible to a numpy array!
    :returns: a 3x3 or 4x4 numpy array based on the input matrix or an empty list otherwise
    """
    v = np.asarray(v)
    rtn = []
    if(v.shape == (6,1)):
        rtn = np.block([[ bracket(v[:3]),  v[3:]   ],
                        [ np.zeros((1,4))          ]])
    elif(v.shape == (3,1)):
        rtn = np.zeros((3,3))
        rtn[0][1] = - v[2]
        rtn[0][2] =   v[1]
        rtn[1][2] = - v[0]
        rtn = rtn - rtn.transpose()
    return rtn

#for solving skew symmetric Matrix...miscellenous function...
def skew4(V_b):
    return np.array([[0,-1*V_b[2],V_b[1],V_b[3]],[V_b[2],0,-1*V_b[0],V_b[4]],[-1*V_b[1],V_b[0],0,V_b[5]],[0,0,0,0]])


#compute the Screw Matrix...
def ToScrew(a, q=None):
    a= np.atleast_2d(a).reshape((3,1))
    if q  is not None:
         q = np.atleast_2d(q).reshape((3,1))
         return np.block([[ a                 ],
                         [ bracket(q).dot(a) ]])
    return np.block([[ np.zeros((3,1)) ],
                     [ a               ]])
def toTs(S, theta): #return 4x4 HCT matrices...
    return [expm(skew4(S[:,i]) * theta[i]) for i in range(S.shape[1])]



#get the joint object Handle...(joint RCM_PSM2)
returnCode1,J1=vrep.simxGetObjectHandle(clientID,"J1_PSM2",vrep.simx_opmode_blocking)
if returnCode1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the J1')
#print("Joint J1_PSM2", J1)


returnCode2,J2=vrep.simxGetObjectHandle(clientID,"J2_PSM2",vrep.simx_opmode_blocking)
if returnCode1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the J2')
#print("Joint J2_PSM2", J2)



returnCode3,J3=vrep.simxGetObjectHandle(clientID,"J3_PSM2",vrep.simx_opmode_blocking)
if returnCode1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the J3')
#print("Joint J3_PSM2", J3)


#get the joint angle of the index finger...(mushua index finger (RCM_PSM2))
#and calulate the kinematics of the index finger...

#Mushua index_1_(RCM_PSM2)
returnCode_ind1, Mu_indJ1=vrep.simxGetObjectHandle(clientID,"mushuahand_joint1_Index",vrep.simx_opmode_blocking)
if returnCode_ind1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint1_Index')
#print("Joint mushuahand_joint1_index",  Mu_indJ1)

#Mushua index_2_(RCM_PSM2)
returnCode_ind2, Mu_indJ2=vrep.simxGetObjectHandle(clientID,"mushuahand_joint2_Index",vrep.simx_opmode_blocking)
if returnCode_ind2 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint2_Index')
#print("Joint mushuahand_joint2_index",  Mu_indJ2)


#Mushua index_3_(RCM_PSM2)
returnCode_ind3, Mu_indJ3=vrep.simxGetObjectHandle(clientID,"mushuahand_joint3_Index",vrep.simx_opmode_blocking)
if returnCode_ind3 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint3_Index')
#print("Joint mushuahand_joint3_index",  Mu_indJ3)


#get the joint angle of the middle finger...(mushua middle)
#calulate the kinematics of the middle finger...
#Mushua middle_1_(RCM_PSM2)
returnCode_mid1, Mu_midJ1=vrep.simxGetObjectHandle(clientID,"mushuahand_joint1_Middle",vrep.simx_opmode_blocking)
if returnCode_mid1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint1_Middle')
#print("Joint mushuahand_joint1_Middle",  Mu_midJ1)

#Mushua middle_2_(RCM_PSM2)
returnCode_mid2, Mu_midJ2=vrep.simxGetObjectHandle(clientID,"mushuahand_joint2_Middle",vrep.simx_opmode_blocking)
if returnCode_mid2 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint2_Middle')
#print("Joint mushuahand_joint2_Middle",  Mu_midJ2)


#Mushua middle_3_(RCM_PSM2)
returnCode_mid3, Mu_midJ3=vrep.simxGetObjectHandle(clientID,"mushuahand_joint3_Middle",vrep.simx_opmode_blocking)
if returnCode_mid3 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint3_Middle')
#print("Joint mushuahand_joint3_Middle",  Mu_midJ3)



#get the joint angle of the thumb finger...(mushua thumb)
#calulate the kinematics of the thumb finger...
#Mushua thumb_1_(RCM_PSM2)
returnCode_thu1, Mu_thuJ1=vrep.simxGetObjectHandle(clientID,"mushuahand_joint1_thumb",vrep.simx_opmode_blocking)
if returnCode_thu1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint1_thumb')
#print("Joint mushuahand_joint1_thumb",  Mu_thuJ1)

#Mushua thumb_2_(RCM_PSM2)
returnCode_thu2, Mu_thuJ2=vrep.simxGetObjectHandle(clientID,"mushuahand_joint2_thumb",vrep.simx_opmode_blocking)
if returnCode_thu2 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint2_thumb')
#print("Joint mushuahand_joint2_thumb",  Mu_thuJ2)


#Mushua thumb_3_(RCM_PSM2)
returnCode_thu3, Mu_thuJ3=vrep.simxGetObjectHandle(clientID,"mushuahand_joint3_thumb",vrep.simx_opmode_blocking)
if returnCode_thu3 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint3_thumb')
#print("Joint mushuahand_joint3_thumb",  Mu_thuJ3)




#get the mushua handle...(RCM_PSM2)
returnCode_handle, mushuaHandle=vrep.simxGetObjectHandle(clientID,"mushuahand_dyn",vrep.simx_opmode_blocking)
if returnCode_handle != vrep.simx_return_ok:
   raise Exception('could not get object handle for the base')
#print("Joint mushuahand_dyn/handle",  mushuaHandle)



#get the mushua base...(RCM_PSM2)
returnCode_base, mushuaBase=vrep.simxGetObjectHandle(clientID,"mushuahand_vis",vrep.simx_opmode_blocking)
if returnCode_base != vrep.simx_return_ok:
   raise Exception('could not get object handle for the base')
#print("Joint mushuahand_vis/base",  mushuaBase)

#get the tip the index fingers tip...
returnCode_ind_tip1, mushuaindtip2=vrep.simxGetObjectHandle(clientID,"mushuahand_Index3",vrep.simx_opmode_blocking)
if returnCode_ind_tip1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_Index3')
#print("Joint mushuahand_Index3/tip",  mushuaindtip2)

#get the tip the middle fingers tip...
returnCode_mid_tip2, mushuamidtip2=vrep.simxGetObjectHandle(clientID,"mushuahand_Middle3",vrep.simx_opmode_blocking)
if returnCode_mid_tip2 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_Middle3')
#print("Joint mushuahand_Middle3/tip",  mushuamidtip2)

#get the tip the thumb fingers tip...
returnCode_thu_tip2, mushuathutip2=vrep.simxGetObjectHandle(clientID,"mushuahand_linkthumb3vis",vrep.simx_opmode_blocking)
if returnCode_thu_tip2 != vrep.simx_return_ok:
   raise Exception('could not get object mushua handle for the mushuathutip2')
#print("Joint mushuahand_linkthumb3vis/tip",  mushuathutip2)

returnCodeDVRKBF, DVRKBF=vrep.simxGetObjectHandle(clientID,"setupJointVisible",vrep.simx_opmode_blocking)
if returnCodeDVRKBF != vrep.simx_return_ok:
   raise Exception('could not get object handle for the DVRKBF')
#print("DVRKBF",  DVRKBF)

returnCodeconn, connection=vrep.simxGetObjectHandle(clientID,"robot_attachment",vrep.simx_opmode_blocking)
if returnCodeconn!= vrep.simx_return_ok:
   raise Exception('could not get object handle for the connection')
#print("robot_attachment",  connection)


returnCodeconn, connection_PSMone=vrep.simxGetObjectHandle(clientID,"robotattachment_1",vrep.simx_opmode_blocking)
if returnCodeconn!= vrep.simx_return_ok:
   raise Exception('could not get object handle for the connection_PSMone')
#print("robot_attachment_1",  connection_PSMone)



#get the joint object Handle...(joint RCM_PSM1)
returnCode1,J1_PSMone=vrep.simxGetObjectHandle(clientID,"J1_PSM1",vrep.simx_opmode_blocking)
if returnCode1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the J1_PSMone')
print("Joint J1_PSM1", J1_PSMone)


returnCode2,J2_PSMone=vrep.simxGetObjectHandle(clientID,"J2_PSM1",vrep.simx_opmode_blocking)
if returnCode1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the J2_PSMone')
print("Joint J2_PSM1", J2_PSMone)



returnCode3,J3_PSMone=vrep.simxGetObjectHandle(clientID,"J3_PSM1",vrep.simx_opmode_blocking)
if returnCode1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the J3_PSMone')
print("Joint J3_PSM1", J3_PSMone)


#get the joint angle of the index finger...(mushua index finger (RCM_PSM2))
#and calulate the kinematics of the index finger...

#Mushua index_1_(RCM_PSM1)
returnCode_ind1, Mu_indJ1_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint1_Index0",vrep.simx_opmode_blocking)
if returnCode_ind1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint1_Index0')
print("Joint mushuahand_joint1_index0",  Mu_indJ1_PSMone)

#Mushua index_2_(RCM_PSM1)
returnCode_ind2, Mu_indJ2_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint2_Index0",vrep.simx_opmode_blocking)
if returnCode_ind2 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint2_Index0')
print("Joint mushuahand_joint2_index0",  Mu_indJ2_PSMone)


#Mushua index_3_(RCM_PSM1)
returnCode_ind3, Mu_indJ3_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint3_Index0",vrep.simx_opmode_blocking)
if returnCode_ind3 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint3_Index0')
print("Joint mushuahand_joint3_index0",  Mu_indJ3_PSMone)


#get the joint angle of the middle finger...(mushua middle)
#calulate the kinematics of the middle finger...
#Mushua middle_1_(RCM_PSM1)
returnCode_mid1, Mu_midJ1_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint1_Middle0",vrep.simx_opmode_blocking)
if returnCode_mid1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint1_Middle0')
#print("Joint mushuahand_joint1_Middle0",  Mu_midJ1_PSMone)

#Mushua middle_2_(RCM_PSM1)
returnCode_mid2, Mu_midJ2_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint2_Middle0",vrep.simx_opmode_blocking)
if returnCode_mid2 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint2_Middle0')
#print("Joint mushuahand_joint2_Middle",  Mu_midJ2)


#Mushua middle_3_(RCM_PSM1)
returnCode_mid3, Mu_midJ3_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint3_Middle0",vrep.simx_opmode_blocking)
if returnCode_mid3 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint3_Middle0')
print("Joint mushuahand_joint3_Middle0",  Mu_midJ3_PSMone)



#get the joint angle of the thumb finger...(mushua thumb)
#calulate the kinematics of the thumb finger...
#Mushua thumb_1_(RCM_PSM1)
returnCode_thu1, Mu_thuJ1_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint1_thumb0",vrep.simx_opmode_blocking)
if returnCode_thu1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint1_thumb0')
print("Joint mushuahand_joint1_thumb",  Mu_thuJ1_PSMone)

#Mushua thumb_2_(RCM_PSM1)
returnCode_thu2, Mu_thuJ2_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint2_thumb0",vrep.simx_opmode_blocking)
if returnCode_thu2 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint2_thumb0')
print("Joint mushuahand_joint2_thumb0",  Mu_thuJ2_PSMone)


#Mushua thumb_3_(RCM_PSM1)
returnCode_thu3, Mu_thuJ3_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_joint3_thumb0",vrep.simx_opmode_blocking)
if returnCode_thu3 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_joint3_thumb0')
print("Joint mushuahand_joint3_thumb0",  Mu_thuJ3_PSMone)




#get the mushua handle...(RCM_PSM1)
returnCode_handle, mushuaHandle_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_vis0",vrep.simx_opmode_blocking)
if returnCode_handle != vrep.simx_return_ok:
   raise Exception('could not get object handle for the base_PSMone')
print("Joint mushuahand_dyn/handle_PSMone",  mushuaHandle_PSMone)



#get the mushua base...(RCM_PSM1)
returnCode_base, mushuaBase_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_vis1",vrep.simx_opmode_blocking)
if returnCode_base != vrep.simx_return_ok:
   raise Exception('could not get object handle for the base_PSMone')
print("Joint mushuahand_vis/base_PSMone",  mushuaBase_PSMone)

#get the tip the index fingers tip...
returnCode_ind_tip1, mushuaindtip2_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_Index5",vrep.simx_opmode_blocking)
if returnCode_ind_tip1 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_Index5')
print("Joint mushuahand_Index3/tip_PSMone",  mushuaindtip2_PSMone)

#get the tip the middle fingers tip...
returnCode_mid_tip2, mushuamidtip2_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_Middle5",vrep.simx_opmode_blocking)
if returnCode_mid_tip2 != vrep.simx_return_ok:
   raise Exception('could not get object handle for the mushuahand_Middle5')
print("Joint mushuahand_Middle3/tip_PSMone",  mushuamidtip2_PSMone)

#get the tip the thumb fingers tip...
returnCode_thu_tip2, mushuathutip2_PSMone=vrep.simxGetObjectHandle(clientID,"mushuahand_linkthumb3vis0",vrep.simx_opmode_blocking)
if returnCode_thu_tip2 != vrep.simx_return_ok:
   raise Exception('could not get object mushua handle for the base_PSMone')
#print("Joint mushuahand_linkthumb3vis/tip",  mushuathutip2_PSMone)

#compute matrix for PSM1...
def ScrewMatrix_PSM1():
    #get the Joint Position of the first 3 DVRK joint 1...3 for the PSM...
    qreturn1,q1_PSMone=vrep.simxGetObjectPosition(clientID,J1_PSMone,DVRKBF,vrep.simx_opmode_blocking)
    if qreturn1 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the J1_PSMone')

    q1_PSMone = np.reshape(q1_PSMone,(3,1))
    a1_PSMone=np.array([[0],[0],[-1]])
    s1_PSMone = ToScrew(a1_PSMone, q1_PSMone)
    print("s1_PSMone",s1_PSMone)

    qreturn2,q2_PSMone=vrep.simxGetObjectPosition(clientID,J2_PSMone,DVRKBF,vrep.simx_opmode_blocking)
    if qreturn2 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the J2_PSMone')

    q2_PSMone = np.reshape(q2_PSMone,(3,1))
    #print(q2)
    a2_PSMone=np.array([[0],[0],[-1]])
    s2_PSMone = ToScrew(a2_PSMone, q2_PSMone)
    #print("s2_PSMone",s2_PSMone)

    qreturn3,q3_PSMone=vrep.simxGetObjectPosition(clientID,J3_PSMone,DVRKBF,vrep.simx_opmode_blocking)
    if qreturn3 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the J3_PSMone')

    q3_PSMone = np.reshape(q3_PSMone,(3,1))
    a3_PSMone=np.array([[0],[0],[1]])
    s3_PSMone = ToScrew(a3_PSMone, q3_PSMone)
    #print("s3_PSMone",s3_PSMone)

    
    
    
    
    #Now lets get the joint position of the Mushua index hand...
    qreturn4, index_qJ1_PSMone=vrep.simxGetObjectPosition(clientID,Mu_indJ1_PSMone,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn4 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_indJ1_PSMone')
    index_qJ1_PSMone = np.reshape(index_qJ1_PSMone,(3,1))
    a_indQJ1_PSMone= np.array([[0],[0],[1]])
    s_indQJ1_PSMone= ToScrew(a_indQJ1_PSMone, index_qJ1_PSMone)
    #print(" s_indQJ1_PSMone", s_indQJ1_PSMone)
    
    qreturn5, index_qJ2_PSMone=vrep.simxGetObjectPosition(clientID,Mu_indJ2_PSMone,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn5 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_indJ2_PSMone')
    index_qJ2_PSMone = np.reshape(index_qJ2_PSMone,(3,1))
    a_indQJ2_PSMone= np.array([[0],[0],[1]])
    s_indQJ2_PSMone= ToScrew(a_indQJ2_PSMone, index_qJ2_PSMone)
    #print(" s_indQJ2_PSMone", s_indQJ2_PSMone)

  
    qreturn6, index_qJ3_PSMone=vrep.simxGetObjectPosition(clientID,Mu_indJ3_PSMone,mushuaHandle_PSMone,vrep.simx_opmode_blocking)
    if qreturn5 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_indJ2_PSMone')
    index_qJ3 = np.reshape(index_qJ3_PSMone,(3,1))
    a_indQJ3_PSMone= np.array([[0],[0],[1]])
    s_indQJ3_PSMone= ToScrew(a_indQJ3_PSMone, index_qJ3_PSMone)
    #print(" s_indQJ3_PSMone", s_indQJ3_PSMone)

    #Now lets get the joint position of the Mushua middle hand...
   
    qreturn7, mid_qJ1_PSMone=vrep.simxGetObjectPosition(clientID,Mu_midJ1_PSMone,mushuaHandle_PSMone,vrep.simx_opmode_blocking)
    if qreturn7 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_midJ1_PSMone')
    mid_qJ1_PSMone = np.reshape(mid_qJ1_PSMone,(3,1))
    a_midQJ1_PSMone= np.array([[0],[0],[1]])
    s_midQJ1_PSMone= ToScrew(a_midQJ1_PSMone, mid_qJ1_PSMone)
    #print(" s_midQJ1_PSMone", s_midQJ1_PSMone)
    
    qreturn8, mid_qJ2_PSMone=vrep.simxGetObjectPosition(clientID,Mu_midJ2_PSMone,mushuaHandle_PSMone,vrep.simx_opmode_blocking)
    if qreturn8 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_midJ2_PSMone')
    mid_qJ2_PSMone = np.reshape(mid_qJ2_PSMone,(3,1))
    a_midQJ2_PSMone= np.array([[0],[0],[1]])
    s_midQJ2_PSMone= ToScrew(a_midQJ2_PSMone, mid_qJ2_PSMone)
    #print("s_midQJ2", s_midQJ2)

    qreturn9, mid_qJ3_PSMone=vrep.simxGetObjectPosition(clientID,Mu_midJ3_PSMone,mushuaHandle_PSMone,vrep.simx_opmode_blocking)
    if qreturn9 != vrep.simx_return_ok:
       raise Exception('could not get object position for the Mu_midJ3_PSMone')
    mid_qJ3_PSMone = np.reshape(mid_qJ3_PSMone,(3,1))
    a_midQJ3_PSMone= np.array([[0],[0],[1]])
    s_midQJ3_PSMone= ToScrew(a_midQJ3_PSMone, mid_qJ3_PSMone)
    #print("s_midQJ3", s_midQJ3)

    #Now lets get the joint position of the Mushua thumb hand...
    
    qreturn10, thu_qJ1_PSMone=vrep.simxGetObjectPosition(clientID, Mu_thuJ1_PSMone,mushuaHandle_PSMone,vrep.simx_opmode_blocking)
    if qreturn10 != vrep.simx_return_ok:
       raise Exception('could not get object position for the Mu_thuJ1_PSMone')
    thu_qJ1_PSMone = np.reshape(thu_qJ1_PSMone,(3,1))
    a_thuQJ1_PSMone= np.array([[0],[0],[1]])
    s_thuQJ1_PSMone= ToScrew(a_thuQJ1_PSMone, thu_qJ1_PSMone )
    #print("  s_thuQJ1_PSMone", s_thuQJ1_PSMone)
    
    
    qreturn11, thu_qJ2_PSMone=vrep.simxGetObjectPosition(clientID, Mu_thuJ3_PSMone,mushuaHandle_PSMone,vrep.simx_opmode_blocking)
    if qreturn11 != vrep.simx_return_ok:
       raise Exception('could not get object position for the Mu_thuJ2_PSMone')
    thu_qJ2_PSMone = np.reshape(thu_qJ2_PSMone,(3,1))
    a_thuQJ2_PSMone= np.array([[0],[0],[1]])
    s_thuQJ2_PSMone= ToScrew(a_thuQJ2_PSMone, thu_qJ2_PSMone )
    #print("  s_thuQJ2_PSMone", s_thuQJ2_PSMone)
    
    qreturn12, thu_qJ3_PSMone=vrep.simxGetObjectPosition(clientID, Mu_thuJ3_PSMone,mushuaHandle_PSMone,vrep.simx_opmode_blocking)
    if qreturn12 != vrep.simx_return_ok:
       raise Exception('could not get object position for the Mu_thuJ2_PSMone')
    thu_qJ3_PSMone = np.reshape(thu_qJ2_PSMone,(3,1))
    a_thuQJ3_PSMone= np.array([[0],[0],[1]])
    s_thuQJ3_PSMone= ToScrew(a_thuQJ3_PSMone, thu_qJ3_PSMone )
    #print(" s_thuQJ3", s_thuQJ3)
    # s_indQJ1,  s_indQJ2  , s_indQJ3 ,  s_midQJ1,   s_midQJ2,  s_midQJ3, s_thuQJ1, s_thuQJ2,  s_thuQJ3

    S_PSM1 = np.block([[s1_PSMone, s2_PSMone, s3_PSMone]])  
    S_mushua1=np.block([[s_indQJ1_PSMone,  s_indQJ2_PSMone  , s_indQJ3_PSMone ,  s_midQJ1_PSMone,   s_midQJ2_PSMone,  s_midQJ3_PSMone, s_thuQJ1_PSMone, s_thuQJ2_PSMone,  s_thuQJ3_PSMone]])
    #print(S[ ,2:4 ])
    #print(len(S))
    return S_PSM1, S_mushua1


def computeThetas_PSM1():
    result1, theta1_PSMone = vrep.simxGetJointPosition(clientID, J1_PSMone, vrep.simx_opmode_blocking)
    if result1 != vrep.simx_return_ok:
        raise Exception('could not get first joint_PSMone variable')
    print('current value of first joint variable on J1_PSMone: theta = {:f}'.format(theta1_PSMone))
   
    result2, theta2_PSMone = vrep.simxGetJointPosition(clientID, J2_PSMone, vrep.simx_opmode_blocking)
    if result2 != vrep.simx_return_ok:
        raise Exception('could not get second joint_PSMone variable')
    print('current value of second joint variable on J2_PSMone: theta = {:f}'.format(theta2_PSMone))

    result3, theta3_PSMone = vrep.simxGetJointPosition(clientID, J3_PSMone, vrep.simx_opmode_blocking)
    if result3 != vrep.simx_return_ok:
        raise Exception('could not get third joint_PSMone variable')
    print('current value of third joint variable on J3: theta = {:f}'.format(theta3_PSMone))
    #theta for the mushua index finger...1-3
    
    #global Mu_indJ1, Mu_indJ2, Mu_indJ3   #Joints index
    result4, indtheta1_PSMone = vrep.simxGetJointPosition(clientID, Mu_indJ1_PSMone, vrep.simx_opmode_blocking)
    if result4 != vrep.simx_return_ok:
        raise Exception('could not get first index joint_PSMone variable')
    print('current value of first joint variable on index finger of the mushua finger Mu_indJ1: theta = {:f}'.format(indtheta1_PSMone))

    result5, indtheta2_PSMone = vrep.simxGetJointPosition(clientID, Mu_indJ2_PSMone, vrep.simx_opmode_blocking)
    if result5 != vrep.simx_return_ok:
        raise Exception('could not get second index joint_PSMone variable')
    print('current value of second joint variable on the index finger of the mushua finger Mu_indJ2_PSMone: theta = {:f}'.format(indtheta2_PSMone))

    result6, indtheta3_PSMone = vrep.simxGetJointPosition(clientID, Mu_indJ3_PSMone, vrep.simx_opmode_blocking)
    if result6 != vrep.simx_return_ok:
        raise Exception('could not get third index joint_PSMone variable')
    print('current value of third joint variable on index finger of the mushua finger Mu_indJ3_PSMone: theta = {:f}'.format(indtheta3_PSMone))

     


    #theta for mushua middle finger
    
     #global Mu_midJ1, Mu_midJ2, Mu_midJ3  #Joints middle

    result7, midtheta1_PSMone = vrep.simxGetJointPosition(clientID, Mu_midJ1_PSMone, vrep.simx_opmode_blocking)
    if result7 != vrep.simx_return_ok:
        raise Exception('could not get first joint_PSMone variable of middle finger one')
    print('current value of first joint variable of the middle finger one.. Mu_midJ1: theta = {:f}'.format(midtheta1_PSMone))

    result8, midtheta2_PSMone = vrep.simxGetJointPosition(clientID, Mu_midJ2_PSMone, vrep.simx_opmode_blocking)
    if result8 != vrep.simx_return_ok:
        raise Exception('could not get second joint_PSMone variable middle finger two')
    print('current value of second joint variable on middle finger two Mu_midJ2_PSMone: theta = {:f}'.format(midtheta2_PSMone))

    result9, midtheta3_PSMone = vrep.simxGetJointPosition(clientID, Mu_midJ3_PSMone, vrep.simx_opmode_blocking)
    if result9 != vrep.simx_return_ok:
        raise Exception('could not get third joint_PSMone variable middle finger three')
    print('current value of third joint_PSMone mushua middle finger three variable on Mu_midJ3_PSMone: theta = {:f}'.format(midtheta2_PSMone))
     
     


    #theta for the thumb finger joint...
    #global Mu_thuJ1, Mu_thuJ2, Mu_thuJ3  #Joints thumb
    result10, thutheta1_PSMone = vrep.simxGetJointPosition(clientID, Mu_thuJ1_PSMone, vrep.simx_opmode_blocking)
    if result10 != vrep.simx_return_ok:
        raise Exception('could not get first joint thumb finger variable of middle finger one')
    print('current value of first joint variable of the thumb finger one.. Mu_midJ1_PSMone: theta = {:f}'.format(thutheta1_PSMone))

    result11, thutheta2_PSMone = vrep.simxGetJointPosition(clientID,Mu_thuJ2_PSMone, vrep.simx_opmode_blocking)
    if result11 != vrep.simx_return_ok:
        raise Exception('could not get second joint_PSMone variable thumb finger two')
    print('current value of second joint variable on thumb finger two Mu_midJ2: theta = {:f}'.format(thutheta2_PSMone))

    result12, thutheta3_PSMone = vrep.simxGetJointPosition(clientID, Mu_thuJ3_PSMone, vrep.simx_opmode_blocking)
    if result12 != vrep.simx_return_ok:
        raise Exception('could not get third joint_PSMone variable thumb finger three')
    print('current value of third joint mushua thumb finger three variable on Mu_midJ3_PSMone: theta = {:f}'.format(thutheta3_PSMone))

     #theta1, theta2, theta3, indtheta1, indtheta2, indtheta3, midtheta1, midtheta2, midtheta3, thutheta1, thutheta2, thutheta3
    theta_PSMone =  np.array([[theta1_PSMone], [theta2_PSMone], [theta3_PSMone]]) 
    #theta = np.array([[indtheta1_PSMone], [indtheta2_PSMone], [indtheta3_PSMone], [midtheta1_PSMone], [midtheta2_PSMone], [midtheta3_PSMone], [thutheta1_PSMone], [thutheta2_PSMone], [thutheta3_PSMone] ])
    return theta_PSMone




#compute the Matrix...
def computeM():
    result, orientation = vrep.simxGetObjectOrientation(clientID, connection, DVRKBF, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object orientation angles for BF')

    # Get the position from base to world frame
    result, p = vrep.simxGetObjectPosition(clientID,connection, DVRKBF, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object current position for BF')
    P_initial = np.reshape(p,(3,1))
    # print ("P_initial", P_initial)
    R_initial = transforms3d.euler.euler2mat(orientation[0], orientation[1], orientation[2])
    # print ("R_itinial", R_initial)
    M = np.block([
    [R_initial[0,:], P_initial[0,:]],
    [R_initial[1,:], P_initial[1,:]],
    [R_initial[2,:], P_initial[2,:]],
    [0,0,0,1] ])
    #print (np.matrix(M))
    return M

def computeMushua():
    result, orientation = vrep.simxGetObjectOrientation(clientID, mushuaindtip2, mushuaHandle, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object orientation angles for BF')

    # Get the position from base to world frame
    result, p = vrep.simxGetObjectPosition(clientID,mushuaindtip2, mushuaHandle, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object current position for BF')
    P_initial = np.reshape(p,(3,1))
    # print ("P_initial", P_initial)
    R_initial = transforms3d.euler.euler2mat(orientation[0], orientation[1], orientation[2])
    # print ("R_itinial", R_initial)
    M = np.block([
    [R_initial[0,:], P_initial[0,:]],
    [R_initial[1,:], P_initial[1,:]],
    [R_initial[2,:], P_initial[2,:]],
    [0,0,0,1] ])
    #print (np.matrix(M))
    return M


def ScrewMatrix():
    #get the Joint Position of the first 3 DVRK joint 1...3 for the PSM...
    qreturn1,q1=vrep.simxGetObjectPosition(clientID,J1,DVRKBF,vrep.simx_opmode_blocking)
    if qreturn1 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the J1')

    q1 = np.reshape(q1,(3,1))
    a1=np.array([[0],[0],[-1]])
    s1 = ToScrew(a1, q1)
    #print("s1",s1)

    qreturn2,q2=vrep.simxGetObjectPosition(clientID,J2,DVRKBF,vrep.simx_opmode_blocking)
    if qreturn2 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the J2')

    q2 = np.reshape(q2,(3,1))
    #print(q2)
    a2=np.array([[0],[0],[-1]])
    s2 = ToScrew(a2, q2)
    #print("s2",s2)

    qreturn3,q3=vrep.simxGetObjectPosition(clientID,J3,DVRKBF,vrep.simx_opmode_blocking)
    if qreturn3 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the J3')

    q3 = np.reshape(q3,(3,1))
    a3=np.array([[0],[0],[1]])
    s3 = ToScrew(a3, q3)
    #print("s3",s3)

    
    
    
    
    #Now lets get the joint position of the Mushua index hand...
    qreturn4, index_qJ1=vrep.simxGetObjectPosition(clientID,Mu_indJ1,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn4 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_indJ1')
    index_qJ1 = np.reshape(index_qJ1,(3,1))
    a_indQJ1= np.array([[0],[0],[1]])
    s_indQJ1= ToScrew(a_indQJ1, index_qJ1)
    #print(" s_indQJ1", s_indQJ1)
    
    qreturn5, index_qJ2=vrep.simxGetObjectPosition(clientID,Mu_indJ2,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn5 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_indJ2')
    index_qJ2 = np.reshape(index_qJ2,(3,1))
    a_indQJ2= np.array([[0],[0],[1]])
    s_indQJ2= ToScrew(a_indQJ2, index_qJ2)
    #print(" s_indQJ2", s_indQJ2)

  
    qreturn6, index_qJ3=vrep.simxGetObjectPosition(clientID,Mu_indJ3,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn5 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_indJ2')
    index_qJ3 = np.reshape(index_qJ3,(3,1))
    a_indQJ3= np.array([[0],[0],[1]])
    s_indQJ3= ToScrew(a_indQJ3, index_qJ3)
    #print(" s_indQJ3", s_indQJ3)

    #Now lets get the joint position of the Mushua middle hand...
   
    qreturn7, mid_qJ1=vrep.simxGetObjectPosition(clientID,Mu_midJ1,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn7 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_midJ1')
    mid_qJ1 = np.reshape(mid_qJ1,(3,1))
    a_midQJ1= np.array([[0],[0],[1]])
    s_midQJ1= ToScrew(a_midQJ1, mid_qJ1)
    #print(" s_midQJ1", s_midQJ1)
    
    qreturn8, mid_qJ2=vrep.simxGetObjectPosition(clientID,Mu_midJ2,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn8 != vrep.simx_return_ok:
       raise Exception('could not get object handle for the Mu_midJ2')
    mid_qJ2 = np.reshape(mid_qJ2,(3,1))
    a_midQJ2= np.array([[0],[0],[1]])
    s_midQJ2= ToScrew(a_midQJ2, mid_qJ2)
    #print("s_midQJ2", s_midQJ2)

    qreturn9, mid_qJ3=vrep.simxGetObjectPosition(clientID,Mu_midJ3,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn9 != vrep.simx_return_ok:
       raise Exception('could not get object position for the Mu_midJ3')
    mid_qJ3 = np.reshape(mid_qJ3,(3,1))
    a_midQJ3= np.array([[0],[0],[1]])
    s_midQJ3= ToScrew(a_midQJ3, mid_qJ3)
    #print("s_midQJ3", s_midQJ3)

    #Now lets get the joint position of the Mushua thumb hand...
    
    qreturn10, thu_qJ1=vrep.simxGetObjectPosition(clientID, Mu_thuJ1,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn10 != vrep.simx_return_ok:
       raise Exception('could not get object position for the Mu_thuJ1')
    thu_qJ1 = np.reshape(thu_qJ1,(3,1))
    a_thuQJ1= np.array([[0],[0],[1]])
    s_thuQJ1= ToScrew(a_thuQJ1, thu_qJ1 )
    #print("  s_thuQJ1", s_thuQJ1)
    
    
    qreturn11, thu_qJ2=vrep.simxGetObjectPosition(clientID, Mu_thuJ3,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn11 != vrep.simx_return_ok:
       raise Exception('could not get object position for the Mu_thuJ2')
    thu_qJ2 = np.reshape(thu_qJ2,(3,1))
    a_thuQJ2= np.array([[0],[0],[1]])
    s_thuQJ2= ToScrew(a_thuQJ2, thu_qJ2 )
    #print("  s_thuQJ2", s_thuQJ2)
    
    qreturn12, thu_qJ3=vrep.simxGetObjectPosition(clientID, Mu_thuJ3,mushuaHandle,vrep.simx_opmode_blocking)
    if qreturn12 != vrep.simx_return_ok:
       raise Exception('could not get object position for the Mu_thuJ2')
    thu_qJ3 = np.reshape(thu_qJ2,(3,1))
    a_thuQJ3= np.array([[0],[0],[1]])
    s_thuQJ3= ToScrew(a_thuQJ3, thu_qJ3 )
    #print(" s_thuQJ3", s_thuQJ3)
    # s_indQJ1,  s_indQJ2  , s_indQJ3 ,  s_midQJ1,   s_midQJ2,  s_midQJ3, s_thuQJ1, s_thuQJ2,  s_thuQJ3

    S_PSM = np.block([[s1, s2, s3]])  
    S_mushua=np.block([[s_indQJ1,  s_indQJ2  , s_indQJ3 ,  s_midQJ1,   s_midQJ2,  s_midQJ3, s_thuQJ1, s_thuQJ2,  s_thuQJ3]])
    #print(S[ ,2:4 ])
    #print(len(S))
    return S_PSM, S_mushua

deg = 180/PI
rad = PI/180
def computeThetas():
    result1, theta1 = vrep.simxGetJointPosition(clientID, J1, vrep.simx_opmode_blocking)
    if result1 != vrep.simx_return_ok:
        raise Exception('could not get first joint variable')
    #print('current value of first joint variable on J1: theta = {:f}'.format(theta1))
   
    result2, theta2 = vrep.simxGetJointPosition(clientID, J2, vrep.simx_opmode_blocking)
    if result2 != vrep.simx_return_ok:
        raise Exception('could not get second joint variable')
    #print('current value of second joint variable on J2: theta = {:f}'.format(theta2))

    result3, theta3 = vrep.simxGetJointPosition(clientID, J3, vrep.simx_opmode_blocking)
    if result3 != vrep.simx_return_ok:
        raise Exception('could not get third joint variable')
    #print('current value of third joint variable on J3: theta = {:f}'.format(theta3))
    #theta for the mushua index finger...1-3
    
    #global Mu_indJ1, Mu_indJ2, Mu_indJ3   #Joints index
    result4, indtheta1 = vrep.simxGetJointPosition(clientID, Mu_indJ1, vrep.simx_opmode_blocking)
    if result4 != vrep.simx_return_ok:
        raise Exception('could not get first joint variable')
    #print('current value of first joint variable on index finger of the mushua finger Mu_indJ1: theta = {:f}'.format(indtheta1))

    result5, indtheta2 = vrep.simxGetJointPosition(clientID, Mu_indJ2, vrep.simx_opmode_blocking)
    if result5 != vrep.simx_return_ok:
        raise Exception('could not get second joint variable')
    #print('current value of second joint variable on the index finger of the mushua finger Mu_indJ2: theta = {:f}'.format(indtheta2))

    result6, indtheta3 = vrep.simxGetJointPosition(clientID, Mu_indJ3, vrep.simx_opmode_blocking)
    if result6 != vrep.simx_return_ok:
        raise Exception('could not get third joint variable')
    #print('current value of third joint variable on index finger of the mushua finger Mu_indJ3: theta = {:f}'.format(indtheta3))

     


    #theta for mushua middle finger
    
     #global Mu_midJ1, Mu_midJ2, Mu_midJ3  #Joints middle

    result7, midtheta1 = vrep.simxGetJointPosition(clientID, Mu_midJ1, vrep.simx_opmode_blocking)
    if result7 != vrep.simx_return_ok:
        raise Exception('could not get first joint variable of middle finger one')
    #print('current value of first joint variable of the middle finger one.. Mu_midJ1: theta = {:f}'.format(midtheta1))

    result8, midtheta2 = vrep.simxGetJointPosition(clientID, Mu_midJ2, vrep.simx_opmode_blocking)
    if result8 != vrep.simx_return_ok:
        raise Exception('could not get second joint variable middle finger two')
    #print('current value of second joint variable on middle finger two Mu_midJ2: theta = {:f}'.format(midtheta2))

    result9, midtheta3 = vrep.simxGetJointPosition(clientID, Mu_midJ3, vrep.simx_opmode_blocking)
    if result9 != vrep.simx_return_ok:
        raise Exception('could not get third joint variable middle finger three')
    #print('current value of third joint mushua middle finger three variable on Mu_midJ3: theta = {:f}'.format(midtheta2))
     
     


    #theta for the thumb finger joint...
    #global Mu_thuJ1, Mu_thuJ2, Mu_thuJ3  #Joints thumb
    result10, thutheta1 = vrep.simxGetJointPosition(clientID, Mu_thuJ1, vrep.simx_opmode_blocking)
    if result10 != vrep.simx_return_ok:
        raise Exception('could not get first joint thumb finger variable of middle finger one')
    #print('current value of first joint variable of the thumb finger one.. Mu_midJ1: theta = {:f}'.format(thutheta1))

    result11, thutheta2 = vrep.simxGetJointPosition(clientID,Mu_thuJ2, vrep.simx_opmode_blocking)
    if result11 != vrep.simx_return_ok:
        raise Exception('could not get second joint variable thumb finger two')
    #print('current value of second joint variable on thumb finger two Mu_midJ2: theta = {:f}'.format(thutheta2))

    result12, thutheta3 = vrep.simxGetJointPosition(clientID, Mu_thuJ3, vrep.simx_opmode_blocking)
    if result12 != vrep.simx_return_ok:
        raise Exception('could not get third joint variable thumb finger three')
    #print('current value of third joint mushua thumb finger three variable on Mu_midJ3: theta = {:f}'.format(thutheta3))

     #theta1, theta2, theta3, indtheta1, indtheta2, indtheta3, midtheta1, midtheta2, midtheta3, thutheta1, thutheta2, thutheta3
    theta =  np.array([[theta1], [theta2], [theta3]]) 
    #theta = np.array([[indtheta1*deg], [indtheta2*deg], [indtheta3*deg], [midtheta1*deg], [midtheta2*deg], [midtheta3*deg], [thutheta1*deg], [thutheta2*deg], [thutheta3*deg] ])
    return theta



# compute Jacobian ...
def Jacobian(S, theta):
    T = toTs(S, theta)
    J = S[:,[0]]
    for i in range(1, S.shape[1]):
        col = T[0]
        for j in range(1, i):
            col = col.dot(T[j])
        newterm = adj_T(col).dot(S[:,[i]])
        J = np.concatenate((J,newterm),axis=1)
    return J

#compute the forward kinematics..
def forwardKinematics(S, theta, M):
    ret = np.identity(4)
    for t in toTs(S, theta):
        ret = ret.dot(t)
    return ret.dot(M)
def adj_T(T):
    """
    Returns the adjoint transformation matrix of T
    :param T: the pose whose 6x6 adjoint matrix to return
    """
    rot, pos = fromPose(T)
    return np.block([[ rot,                   np.zeros((3,3)) ],
                     [ bracket(pos).dot(rot), rot             ]])

def fromPose(T):
    """
    Returns a rotation matrix and position vector from a 4x4 HCT matrix
    :param T: The 4x4 HCT matrix as either python lists or numpy array
    :returns: a tuple with the first element being a 3x3 numpy array representing the rotation matrix
              and the second element being a 3x1 numpy array position vector
    """
    T = np.asarray(T)
    return (T[:3,:3], T[:3, 3:4])
#lets get some input from mouse...
global x, y
def user_input():
    startTime=time.time()
    print("let get mouse x and y input")
    vrep.simxGetIntegerParameter(clientID,vrep.sim_intparam_mouse_x,vrep.simx_opmode_streaming) # Initialize streaming
    vrep.simxGetIntegerParameter(clientID,vrep.sim_intparam_mouse_y,vrep.simx_opmode_streaming)
    while time.time()-startTime < 5:
	  returnCode,datax=vrep.simxGetIntegerParameter(clientID,vrep.sim_intparam_mouse_x,vrep.simx_opmode_streaming)
	  returnCode,datay=vrep.simxGetIntegerParameter(clientID,vrep.sim_intparam_mouse_y,vrep.simx_opmode_streaming)
	  if returnCode==vrep.simx_return_ok: 
	     x = float(datax)
	     y = float(datay)
             print ("mouse x data",x)
             print("mouse y data", y)
	     print ("Let's keep the Z translation position at the floor (0)")
		    # z = float(input("Enter Z translation position: "))
		    # a = float(input("Enter a rotational angle in degrees: "))
		    # b = float(input("Enter b rotational angle in degrees: "))
		    # c = float(input("Enter c rotational angle in degrees: "))
	     z = 0
	     a = 0
	     b = 0
	     c = 45
	     Goal_pose = RotationMatrixToPose(x, y, z, a, b, c)
	     return Goal_pose




def deg2rad(deg):
    return deg * (np.pi) / 180


def rad2deg(rad):
    return rad * 180 / (np.pi)


def RotationMatrixToPose(x, y, z, a, b, c):
    Goal_pose = np.zeros((4,4))
    Goal_pose[0,3] = x
    Goal_pose[1,3] = y
    Goal_pose[2,3] = z
    Goal_pose[3,3] = 1

    Rot_x = np.array([[1, 0, 0],
                      [0, math.cos(deg2rad(a)), -1*math.sin(deg2rad(a))],
                      [0, math.sin(deg2rad(a)), math.cos(deg2rad(a))]])

    Rot_y = np.array([[math.cos(deg2rad(b)), 0, math.sin(deg2rad(b))],
                      [0, 1, 0],
                      [-1*math.sin(deg2rad(b)), 0, math.cos(deg2rad(b))]])

    Rot_z = np.array([[math.cos(deg2rad(c)), -1*math.sin(deg2rad(c)), 0],
                      [math.sin(deg2rad(c)), math.cos(deg2rad(c)), 0],
                      [0, 0, 1]])


    R = multi_dot([Rot_x, Rot_y, Rot_z])
    Goal_pose[0:3,0:3] = R
    return Goal_pose

#I compute the inverse kinematics using the Newton method using the above parameters...
def invkinematics(endT, S, M, theta=None, max_iter=100, max_err = 0.001, mu=0.05):
    if  theta is None:
        theta = np.zeros((S.shape[1],1))
    V = np.ones((6,1))
    while np.linalg.norm(V) > max_err and max_iter > 0:
        curr_pose = forwardKinematics(S, theta, M)
        #print(curr_pose)
        V = inv_bracket(logm(endT.dot(inv(curr_pose))))
        J = Jacobian(S, theta)
        pinv = inv(J.transpose().dot(J) + mu*np.identity(S.shape[1])).dot(J.transpose())
        thetadot = pinv.dot(V)
        theta = theta + thetadot
        max_iter -= 1;
    return (theta, np.linalg.norm(V))

#print(theta)
#Mushuajaco = Jacobian(S_PSM, theta)
#fk_Mushua = forwardKinematics(S_Mushua, thetaMu, Mushua)
#print("jacobian", np.matrix(Mushuajaco) )
#compute the inverse kinematics...(thetas)

#invthetaspsm2, V =invkinematics(T_2in2, S_PSM, M, theta=None, max_iter=100, max_err = 0.001, mu=0.05)
#print("PSM2", np.matrix(invthetaspsm2))

#invthetasMushua1, V =invkinematics(T_2in0, S_Mushua, Mushua, theta=None, max_iter=100, max_err = 0.001, mu=0.05)
#print("mushua1", np.matrix(invthetasMushua1))
  
#set joint position...(RCM_PSM2)
########################################################################
def set_current_position_callback(msg):
    #set joint point...
    returnCodeposJ1=vrep.simxSetJointPosition(clientID,J1,msg.position[0],vrep.simx_opmode_blocking)
    if returnCodeposJ1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J1_PSM2')
    print("=====posJ1 ====", msg.position[0])
 

    returnCodeposJ2=vrep.simxSetJointPosition(clientID,J2,msg.position[1],vrep.simx_opmode_blocking)
    if returnCodeposJ1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J2_PSM2')
    print("=====posJ2 ====", msg.position[1])

    
    returnCodeposJ3=vrep.simxSetJointPosition(clientID,J3,msg.position[2],vrep.simx_opmode_blocking)
    if returnCodeposJ3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J3_PSM2')
    print("=====posJ3 ====", msg.position[2])

    #set mushua index  finger position  PSM2
    returnCode_posind1=vrep.simxSetJointPosition(clientID,Mu_indJ1,msg.position[3],vrep.simx_opmode_blocking)
    if returnCode_posind1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Index')
    print(" mushua index finger1", msg.position[3])

    returnCode_posind2=vrep.simxSetJointPosition(clientID,Mu_indJ2,msg.position[4],vrep.simx_opmode_blocking)
    if returnCode_posind2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Index')
    print(" mushua index finger2", msg.position[4])

    
    returnCode_posind3=vrep.simxSetJointPosition(clientID,Mu_indJ3,msg.position[5],vrep.simx_opmode_blocking)
    if returnCode_posind3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_Index')
    #print(" mushua index finger3", msg.position[5])


    #set mushua middle finger  position  PSM2
    returnCode_posmid1=vrep.simxSetJointPosition(clientID,Mu_midJ1,msg.position[6],vrep.simx_opmode_blocking)
    if  returnCode_posmid1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Middle')
    print(" mushua Middle finger1", msg.position[6])


    returnCode_posmid2=vrep.simxSetJointPosition(clientID,Mu_midJ2,msg.position[7],vrep.simx_opmode_blocking)
    if  returnCode_posmid2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle')
    print(" mushua Middle finger2", msg.position[7])

    
    returnCode_posmid3=vrep.simxSetJointPosition(clientID,Mu_midJ3,msg.position[8],vrep.simx_opmode_blocking)
    if  returnCode_posmid3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_Middle')
    #print(" mushua Middle finger3", msg.position[8])

    
    #set mushua thumb finger  position  PSM2
    returnCode_posthu1=vrep.simxSetJointPosition(clientID,Mu_thuJ1,msg.position[9],vrep.simx_opmode_blocking)
    if  returnCode_posthu1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_thumb')
    #print(" mushua thumb finger1", msg.position[9])


    returnCode_posthu2=vrep.simxSetJointPosition(clientID,Mu_thuJ2,msg.position[10],vrep.simx_opmode_blocking)
    if  returnCode_posthu2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_thumb')
    print(" mushua thumb finger2", msg.position[10])

    
    returnCode_posthu3=vrep.simxSetJointPosition(clientID,Mu_thuJ3,msg.position[11],vrep.simx_opmode_blocking)
    if  returnCode_posthu3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_thumb')
    print(" mushua thumb finger3", msg.position[11])


def sub_current_pose():
    #rospy.init_node("vrep_catkin_ws", anonymous=True)
    #rospy.loginfo("Alex is just getting started")
    subSetjoint=rospy.Subscriber("/dvrk/PSM2/state_joint_current", JointState, set_current_position_callback, queue_size=1)
    #print(subSetjoint)
    rospy.spin()




def set_desired_position_callback(msg):
    #set joint point...
    #position=[-66.0,0.0,0.0,65.0,60.0, 25.0, -66.0,60.0,21.0]
    returnCodeposJ1=vrep.simxSetJointPosition(clientID,J1,msg.position[0],vrep.simx_opmode_streaming)
    if returnCodeposJ1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J1_PSM2')
    print("=====posJ1 ====", msg.position[0])
 
    returnCodeposJ1=vrep.simxSetJointTargetVelocity(clientID,J1,msg.velocity[0],vrep.simx_opmode_streaming)
    if returnCodeposJ1 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the J1_PSM2')
    print("=====vel1 ====", msg.velocity[0])
   
    returnCodeposJ2=vrep.simxSetJointPosition(clientID,J2,msg.position[1],vrep.simx_opmode_streaming)
    if returnCodeposJ1 != vrep.simx_return_ok:
      raise Exception('could not set object position for the J2_PSM2')
    print("=====posJ2 ====", msg.position[1])

    returnCodeposJ2=vrep.simxSetJointTargetVelocity(clientID,J2,msg.velocity[1],vrep.simx_opmode_streaming)
    if returnCodeposJ2 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the J2_PSM2')
    print("=====vel2 ====", msg.velocity[1])


    returnCodeposJ3=vrep.simxSetJointPosition(clientID,J3,msg.position[2],vrep.simx_opmode_streaming)
    if returnCodeposJ3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J3_PSM2')
    print("=====posJ3 ====", msg.position[2])



    returnCodeposJ3=vrep.simxSetJointTargetVelocity(clientID,J1,msg.velocity[2],vrep.simx_opmode_streaming)
    if returnCodeposJ3 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the J1_PSM2')
    print("=====vel3 ====", msg.velocity[2])

    #set mushua index  finger position  PSM2
    returnCode_posind1=vrep.simxSetJointPosition(clientID,Mu_indJ1, msg.position[3],vrep.simx_opmode_streaming)
    if returnCode_posind1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Index')
    print(" mushua index finger1", msg.position[3])

    returnCode_posind1=vrep.simxSetJointTargetVelocity(clientID,Mu_indJ1, msg.velocity[3],vrep.simx_opmode_streaming)
    if returnCode_posind1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Index')
    print(" mushua index finger1", msg.velocity[3])




    returnCode_posind2=vrep.simxSetJointPosition(clientID,Mu_indJ2,msg.position[4],vrep.simx_opmode_streaming)
    if returnCode_posind2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Index')
    print(" mushua index finger2", msg.position[4])

    returnCode_posind2=vrep.simxSetJointTargetVelocity(clientID,Mu_indJ2, msg.velocity[4],vrep.simx_opmode_streaming)
    if returnCode_posind2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Index')
    print(" mushua index finger1", msg.velocity[4])

    returnCode_posind3=vrep.simxSetJointPosition(clientID,Mu_indJ3, msg.position[5],vrep.simx_opmode_streaming)
    if returnCode_posind3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_Index')
    print(" mushua index finger3", msg.position[5]*rad)
    returnCode_posind3=vrep.simxSetJointTargetVelocity(clientID,Mu_indJ3, msg.velocity[5],vrep.simx_opmode_streaming)
    if returnCode_posind3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Index')
    print(" mushua index finger1", msg.velocity[5])


    #set mushua middle finger  position  PSM2
    returnCode_posmid1=vrep.simxSetJointPosition(clientID,Mu_midJ1,msg.position[6],vrep.simx_opmode_streaming)
    if  returnCode_posmid1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Middle')
    print(" mushua Middle finger1", msg.position[6])

    returnCode_posmid1=vrep.simxSetJointTargetVelocity(clientID,Mu_midJ1,msg.velocity[6],vrep.simx_opmode_streaming)
    if  returnCode_posmid1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Middle')
    print(" mushua Middle finger1", msg.velocity[6])

    returnCode_posmid2=vrep.simxSetJointPosition(clientID,Mu_midJ2,msg.position[7],vrep.simx_opmode_streaming)
    if  returnCode_posmid2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle')
    print(" mushua Middle finger2", msg.position[7])
    returnCode_posmid2=vrep.simxSetJointTargetVelocity(clientID,Mu_midJ2,msg.velocity[7],vrep.simx_opmode_streaming)
    if  returnCode_posmid2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle')
    print(" mushua Middle finger2", msg.velocity[7])
   
    #simxSetJointTargetVelocity 
    
    returnCode_posmid3=vrep.simxSetJointPosition(clientID,Mu_midJ3,msg.position[8],vrep.simx_opmode_streaming)
    if  returnCode_posmid3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_Middle')
    print(" mushua Middle finger3", msg.position[8])

    returnCode_posmid3=vrep.simxSetJointTargetVelocity(clientID,Mu_midJ3,msg.velocity[8],vrep.simx_opmode_streaming)
    if  returnCode_posmid3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle')
    print(" mushua Middle finger3", msg.velocity[8])
    #time.sleep(3)
    #set mushua thumb finger  position  PSM2
    returnCode_posthu1=vrep.simxSetJointPosition(clientID,Mu_thuJ1,msg.position[9],vrep.simx_opmode_streaming)
    if  returnCode_posthu1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_thumb')
    print(" mushua thumb finger1", msg.position[9])
    returnCode_posthu1=vrep.simxSetJointTargetVelocity(clientID,Mu_thuJ1,msg.velocity[9],vrep.simx_opmode_streaming)
    if  returnCode_posthu1!= vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle')
    print(" mushua thumb finger1", msg.velocity[9])
    
    returnCode_posthu2=vrep.simxSetJointPosition(clientID, Mu_thuJ2, msg.position[10], vrep.simx_opmode_streaming)
    if  returnCode_posthu2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_thumb')
    print(" mushua thumb finger2", msg.position[10])

    returnCode_posthu2 = vrep.simxSetJointTargetVelocity(clientID, Mu_thuJ2, msg.velocity[10],vrep.simx_opmode_streaming)
    if  returnCode_posthu2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_thumb')

    print(" mushua thumb finger2", msg.velocity[10])
    returnCode_posthu3=vrep.simxSetJointPosition(clientID,Mu_thuJ3,msg.position[11], vrep.simx_opmode_streaming)
    if  returnCode_posthu3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_thumb')
    print(" mushua thumb finger3", msg.position[11])
    returnCode_posthu2=vrep.simxSetJointTargetVelocity(clientID,Mu_thuJ3,msg.velocity[11],vrep.simx_opmode_streaming)
    if  returnCode_posthu2!= vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle')
    print(" mushua thumb finger3", msg.velocity[11])
    time.sleep(10)
    #vrep.simxFinish(clientID)

#publish desired joint
def publishDesiredjoint():
    Mushua = computeMushua()
    M=computeM()
    S_PSM, S_Mushua=ScrewMatrix()
    theta =computeThetas()
    #T_2in2=  user_input()
    #invthetaspsm2, V =invkinematics(T_2in2, S_PSM, M, theta=None, max_iter=100, max_err = 0.001, mu=0.05)
    pubjoint= rospy.Publisher("/dvrk/PSM2/state_joint_desired", JointState, queue_size=10)
    #pubjoint= rospy.Publisher("/dvrk/PSM1/state_joint_desired", JointState, queue_size=1)
    msg = JointState()
    rate=rospy.Rate(10)
    position=[] #np.array([-0.6981316804885864], [0.2617993950843811], [0.1599999964237213])
    #a = invthetaspsm2
    a= np.array([[-0.6981316804885864], [0.2617993950843811], [0.1599999964237213]]) # invthetaspsm2 #np.array([[39.0],[32.0],[50.0]])
    c=np.append(a, [[-65.555*PI/180],[-0.0*PI/180],[-0.0*PI/180],[65.555*PI/180],[60*PI/180], [-21/PI*180], [-65.555*PI/180],[60*PI/180],[-21*PI/180]])
    position = c
    print(position)
    print("the length of the position",len(position))
    velocity = np.array([[1],[1],[1],[1],[0.1],[0.1],[1.0],[1.0],[-0.1],[-1.0],[1.0],[0.1]])
    #add_velocity= [str(x) for x in add_velocity ]
    #velocity =robvelocity
    print(velocity)
    print("the length of the position",len(velocity))
    msg.position = position
    msg.velocity = velocity 
    while not rospy.is_shutdown():
          pubjoint.publish(msg)
          rate.sleep()


def publishjointdesiredopen():
    
    pubjoint= rospy.Publisher("/dvrk/PSM2/state_joint_desired", JointState, queue_size=10)
    #pubjoint= rospy.Publisher("/dvrk/PSM1/state_joint_desired", JointState, queue_size=1)
    msg = JointState()
    rate=rospy.Rate(10)
    position=[] #np.array([-0.6981316804885864], [0.2617993950843811], [0.1599999964237213])
    a= np.array([[-0.6981316804885864], [0.2617993950843811], [0.1599999964237213]]) # invthetaspsm2 #np.array([[39.0],[32.0],[50.0]])
    c=np.append(a, [[0.000001708*PI/180],[-0.00001131*PI/180],[-0.00001622*PI/180],[0.000005123*PI/180],[-0.000004482*PI/180], [0.00001537*PI/180], [0.000005123*PI/180],[0.000003415*PI/180],[-0.000008538*PI/180]])
    position = c
    print(position)
    print("the length of the position",len(position))
    velocity = np.array([[1],[1],[1],[-1],[-0.1],[-0.1],[-1.0],[-1.0],[0.1],[1.0],[-1.0],[-0.1]])
    #add_velocity= [str(x) for x in add_velocity ]
    #velocity =robvelocity
    print(velocity)
    print("the length of the position",len(velocity))
    msg.position = position
    msg.velocity = velocity 
    while not rospy.is_shutdown():
          pubjoint.publish(msg)
          rate.sleep()





def sub_desired_pose():
    #rospy.init_node("vrep_catkin_ws", anonymous=True)
    #rospy.loginfo("Alex is just getting started")
    subSetjoint=rospy.Subscriber("/dvrk/PSM2/state_joint_desired", JointState, set_desired_position_callback)
    #print(subSetjoint)
    #rospy.spin()
#get the image


#call the current position of the PSM1

def sub_currentPSM1_pose():
    #rospy.init_node("vrep_catkin_ws", anonymous=True)
    #rospy.loginfo("Alex is just getting started")
    subSetjoint=rospy.Subscriber("/dvrk/PSM1/state_joint_current", JointState, set_current_position_callback, queue_size=1)
    #print(subSetjoint)
    rospy.spin()

def set_current_position_callback_PSM1(msg):
    #set joint point...
    returnCodeposJ1=vrep.simxSetJointPosition(clientID,J1_PSMone,msg.position[0],vrep.simx_opmode_blocking)
    if returnCodeposJ1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J1_PSM2')
    print("=====posJ1_PSMone ====", msg.position[0])
 

    returnCodeposJ2=vrep.simxSetJointPosition(clientID,J2_PSMone,msg.position[1],vrep.simx_opmode_blocking)
    if returnCodeposJ1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J2_PSM1')
    print("=====posJ2_PSMone ====", msg.position[1])

    
    returnCodeposJ3=vrep.simxSetJointPosition(clientID,J3_PSMone,msg.position[2],vrep.simx_opmode_blocking)
    if returnCodeposJ3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J3_PSM1')
    print("=====posJ3_PSMone ====", msg.position[2])

    #set mushua index  finger position  PSM1
    returnCode_posind1=vrep.simxSetJointPosition(clientID,Mu_indJ1_PSMone,msg.position[3],vrep.simx_opmode_blocking)
    if returnCode_posind1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Index0')
    print(" mushua  pos index_PSMone finger1", msg.position[3])

    returnCode_posind2=vrep.simxSetJointPosition(clientID,Mu_indJ2_PSMone,msg.position[4],vrep.simx_opmode_blocking)
    if returnCode_posind2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Index0')
    print(" mushua pos index_PSMone finger2", msg.position[4])

    
    returnCode_posind3=vrep.simxSetJointPosition(clientID,Mu_indJ3_PSMone,msg.position[5],vrep.simx_opmode_blocking)
    if returnCode_posind3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_Index0')
    print(" mushua pos index_PSMone finger3", msg.position[5])


    #set mushua middle finger  position  PSM1
    returnCode_posmid1=vrep.simxSetJointPosition(clientID,Mu_midJ1_PSMone,msg.position[6],vrep.simx_opmode_blocking)
    if  returnCode_posmid1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Middle0')
    print(" mushua pos Middle_PSMone finger1", msg.position[6])


    returnCode_posmid2=vrep.simxSetJointPosition(clientID,Mu_midJ2_PSMone,msg.position[7],vrep.simx_opmode_blocking)
    if  returnCode_posmid2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle0')
    print(" mushua pos Middle_PSMone finger2", msg.position[7])

    
    returnCode_posmid3=vrep.simxSetJointPosition(clientID,Mu_midJ3_PSMone,msg.position[8],vrep.simx_opmode_blocking)
    if  returnCode_posmid3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_Middle0')
    print(" mushua pos Middle_PSMone finger3", msg.position[8])

    
    #set mushua thumb finger  position  PSM1
    returnCode_posthu1=vrep.simxSetJointPosition(clientID,Mu_thuJ1_PSMone,msg.position[9],vrep.simx_opmode_blocking)
    if  returnCode_posthu1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_thumb0')
    print(" mushua pos thumb_PSMone finger1", msg.position[9])


    returnCode_posthu2=vrep.simxSetJointPosition(clientID,Mu_thuJ2_PSMone,msg.position[10],vrep.simx_opmode_blocking)
    if  returnCode_posthu2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_thumb0')
    print(" mushua pos thumb_PSMone finger2", msg.position[10])

    
    returnCode_posthu3=vrep.simxSetJointPosition(clientID,Mu_thuJ3_PSMone,msg.position[11],vrep.simx_opmode_blocking)
    if  returnCode_posthu3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_thumb0')
    print(" mushua pos thumb finger3", msg.position[11])


def sub_current_pose_PSM1():
    #rospy.init_node("vrep_catkin_ws", anonymous=True)
    #rospy.loginfo("Alex is just getting started")
    subSetjoint=rospy.Subscriber("/dvrk/PSM1/state_joint_current", JointState, set_current_position_callback_PSM1, queue_size=1)
    #print(subSetjoint)
    rospy.spin()




#desired position callback for PSM1
def set_desired_position_callback_PSM1(msg):
    #set joint point...
    #position=[-66.0,0.0,0.0,65.0,60.0, 25.0, -66.0,60.0,21.0]
    returnCodeposJ1=vrep.simxSetJointPosition(clientID,J1_PSMone,msg.position[0],vrep.simx_opmode_streaming)
    if returnCodeposJ1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J1_PSM1')
    print("=====posJ1_PSMone ====", msg.position[0])
 
    returnCodeposJ1=vrep.simxSetJointTargetVelocity(clientID,J1_PSMone,msg.velocity[0],vrep.simx_opmode_streaming)
    if returnCodeposJ1 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the J1_PSM1')
    print("=====vel1_PSMone ====", msg.velocity[0])
   
    returnCodeposJ2=vrep.simxSetJointPosition(clientID,J2_PSMone,msg.position[1],vrep.simx_opmode_streaming)
    if returnCodeposJ1 != vrep.simx_return_ok:
      raise Exception('could not set object position for the J2_PSM1')
    print("=====posJ2_PSMone ====", msg.position[1])

    returnCodeposJ2=vrep.simxSetJointTargetVelocity(clientID,J2_PSMone,msg.velocity[1],vrep.simx_opmode_streaming)
    if returnCodeposJ2 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the J2_PSM1')
    print("=====vel2_PSMone ====", msg.velocity[1])


    returnCodeposJ3=vrep.simxSetJointPosition(clientID,J3_PSMone,msg.position[2],vrep.simx_opmode_streaming)
    if returnCodeposJ3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the J3_PSM1')
    print("=====posJ3_PSMone ====", msg.position[2])



    returnCodeposJ3=vrep.simxSetJointTargetVelocity(clientID,J3_PSMone,msg.velocity[2],vrep.simx_opmode_streaming)
    if returnCodeposJ3 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the J1_PSM1')
    print("=====vel3_PSMone ====", msg.velocity[2])

    #set mushua index  finger position  PSM2
    returnCode_posind1=vrep.simxSetJointPosition(clientID,Mu_indJ1_PSMone, msg.position[3],vrep.simx_opmode_streaming)
    if returnCode_posind1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Index0')
    print(" mushua index finger1", msg.position[3])

    returnCode_posind1=vrep.simxSetJointTargetVelocity(clientID,Mu_indJ1_PSMone, msg.velocity[3],vrep.simx_opmode_streaming)
    if returnCode_posind1 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the mushuahand_joint1_Index0')
    print(" mushua index0 finger1", msg.velocity[3])




    returnCode_posind2=vrep.simxSetJointPosition(clientID,Mu_indJ2_PSMone,msg.position[4],vrep.simx_opmode_streaming)
    if returnCode_posind2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Index0')
    print(" mushua index0 finger2", msg.position[4])

    returnCode_posind2=vrep.simxSetJointTargetVelocity(clientID,Mu_indJ2_PSMone, msg.velocity[4],vrep.simx_opmode_streaming)
    if returnCode_posind2 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the mushuahand_joint2_Index0')
    print(" mushua index finger1", msg.velocity[4])

    returnCode_posind3=vrep.simxSetJointPosition(clientID,Mu_indJ3_PSMone, msg.position[5],vrep.simx_opmode_streaming)
    if returnCode_posind3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_Index0')
    print(" mushua index finger3", msg.position[5])
    returnCode_posind3=vrep.simxSetJointTargetVelocity(clientID,Mu_indJ3_PSMone, msg.velocity[5],vrep.simx_opmode_streaming)
    if returnCode_posind3 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the mushuahand_joint3_Index0')
    print(" mushua index0 finger1", msg.velocity[5])


    #set mushua middle finger  position  PSM2
    returnCode_posmid1=vrep.simxSetJointPosition(clientID,Mu_midJ1_PSMone,msg.position[6],vrep.simx_opmode_streaming)
    if  returnCode_posmid1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_Middle0')
    print(" mushua Middle0 finger1", msg.position[6])

    returnCode_posmid1=vrep.simxSetJointTargetVelocity(clientID,Mu_midJ1_PSMone,msg.velocity[6],vrep.simx_opmode_streaming)
    if  returnCode_posmid1 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the mushuahand_joint1_Middle0')
    print(" mushua Middle0 finger1", msg.velocity[6])

    returnCode_posmid2=vrep.simxSetJointPosition(clientID,Mu_midJ2_PSMone,msg.position[7],vrep.simx_opmode_streaming)
    if  returnCode_posmid2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle0')
    print(" mushua Middle finger2", msg.position[7])
    returnCode_posmid2=vrep.simxSetJointTargetVelocity(clientID,Mu_midJ2_PSMone,msg.velocity[7],vrep.simx_opmode_streaming)
    if  returnCode_posmid2 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the mushuahand_joint2_Middle')
    print(" mushua Middle0 finger2", msg.velocity[7])
   
    #simxSetJointTargetVelocity 
    
    returnCode_posmid3=vrep.simxSetJointPosition(clientID,Mu_midJ3_PSMone,msg.position[8],vrep.simx_opmode_streaming)
    if  returnCode_posmid3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_Middle0')
    print(" mushua Middle0 finger3", msg.position[8])

    returnCode_posmid3=vrep.simxSetJointTargetVelocity(clientID,Mu_midJ3_PSMone,msg.velocity[8],vrep.simx_opmode_streaming)
    if  returnCode_posmid3 != vrep.simx_return_ok:
       raise Exception('could not set object velocity for the mushuahand_joint2_Middle0')
    print(" mushua Middle0 finger3", msg.velocity[8])
    #time.sleep(3)
    #set mushua thumb finger  position  PSM2
    returnCode_posthu1=vrep.simxSetJointPosition(clientID,Mu_thuJ1_PSMone,msg.position[9],vrep.simx_opmode_streaming)
    if  returnCode_posthu1 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint1_thumb0')
    print(" mushua thumb finger1", msg.position[9])
    returnCode_posthu1=vrep.simxSetJointTargetVelocity(clientID,Mu_thuJ1_PSMone,msg.velocity[9],vrep.simx_opmode_streaming)
    if  returnCode_posthu1!= vrep.simx_return_ok:
       raise Exception('could not set object velocity for the mushuahand_joint2_Middle0')
    print(" mushua thumb0 finger1", msg.velocity[9])
    
    returnCode_posthu2=vrep.simxSetJointPosition(clientID, Mu_thuJ2_PSMone, msg.position[10], vrep.simx_opmode_streaming)
    if  returnCode_posthu2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_thumb0')
    print(" mushua thumb finger2", msg.position[10])

    returnCode_posthu2 = vrep.simxSetJointTargetVelocity(clientID, Mu_thuJ2_PSMone, msg.velocity[10],vrep.simx_opmode_streaming)
    if  returnCode_posthu2 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_thumb0')

    print(" mushua thumb0 finger2", msg.velocity[10])
    returnCode_posthu3=vrep.simxSetJointPosition(clientID,Mu_thuJ3_PSMone,msg.position[11], vrep.simx_opmode_streaming)
    if  returnCode_posthu3 != vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint3_thumb0')
    print(" mushua thumb0 finger3", msg.position[11])
    returnCode_posthu2=vrep.simxSetJointTargetVelocity(clientID,Mu_thuJ3_PSMone,msg.velocity[11],vrep.simx_opmode_streaming)
    if  returnCode_posthu2!= vrep.simx_return_ok:
       raise Exception('could not set object position for the mushuahand_joint2_Middle0')
    print(" mushua thumb0 finger3", msg.velocity[11])
    time.sleep(10)
    #vrep.simxFinish(clientID)

#publish desired joint
def publishDesiredjointPSM1():
   
    pubjoint= rospy.Publisher("/dvrk/PSM1/state_joint_desired", JointState, queue_size=10)
    #pubjoint= rospy.Publisher("/dvrk/PSM1/state_joint_desired", JointState, queue_size=1)
    msg = JointState()
    rate=rospy.Rate(10)
    position=[] #np.array([-0.6981316804885864], [0.2617993950843811], [0.1599999964237213])
    a= np.array([[0.6981316804885864], [0.2617993950843811], [0.1599999964237213]]) # invthetaspsm2 #np.array([[39.0],[32.0],[50.0]])
    c=np.append(a, [[-65.555*PI/180],[-0.0*PI/180],[-0.0*PI/180],[65.555*PI/180],[60*PI/180], [-21/PI*180], [-65.555*PI/180],[60*PI/180],[-21*PI/180]])
    position = c
    print(position)
    print("the length of the position",len(position))
    velocity = np.array([[1],[1],[1],[1],[0.1],[0.1],[1.0],[1.0],[-0.1],[-1.0],[1.0],[0.1]])
    #add_velocity= [str(x) for x in add_velocity ]
    #velocity =robvelocity
    print(velocity)
    print("the length of the position",len(velocity))
    msg.position = position
    msg.velocity = velocity 
    while not rospy.is_shutdown():
          pubjoint.publish(msg)
          rate.sleep()

def publishjointdesiredopen_PSM1():
    
    pubjoint= rospy.Publisher("/dvrk/PSM1/state_joint_desired", JointState, queue_size=10)
    #pubjoint= rospy.Publisher("/dvrk/PSM1/state_joint_desired", JointState, queue_size=1)
    msg = JointState()
    rate=rospy.Rate(10)
    position=[] #np.array([-0.6981316804885864], [0.2617993950843811], [0.1599999964237213])
    a= np.array([[0.6981316804885864], [0.2617993950843811], [0.1599999964237213]]) # invthetaspsm2 #np.array([[39.0],[32.0],[50.0]])
    c=np.append(a, [[0.000005976*PI/180],[0.000008538*PI/180],[0.000001708*PI/180],[0.00001537*PI/180],[-0.000001708*PI/180], [0.00001366*PI/180], [0.000005123*PI/180],[0.000003415*PI/180],[0.000009498*PI/180]])
    position = c
    print(position)
    print("the length of the position",len(position))
    velocity = np.array([[1],[1],[1],[-1],[-0.1],[-0.1],[-1.0],[-1.0],[0.1],[1.0],[-1.0],[-0.1]])
    #add_velocity= [str(x) for x in add_velocity ]
    #velocity =robvelocity
    print(velocity)
    print("the length of the position",len(velocity))
    msg.position = position
    msg.velocity = velocity 
    while not rospy.is_shutdown():
          pubjoint.publish(msg)
          rate.sleep()


def sub_desired_pose_PSM1():
    #rospy.init_node("vrep_catkin_ws", anonymous=True)
    #rospy.loginfo("Alex is just getting started")
    subSetjoint=rospy.Subscriber("/dvrk/PSM1/state_joint_desired", JointState, set_desired_position_callback_PSM1)
    #print(subSetjoint)
    #rospy.spin()
#get the image


def image_callback(data):
    br = CvBridge()
    try:
       cv_image = br.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
       print(e)
    
    (rows, cols, channels) = cv_image.shape
    
    #print([rows, cols,channels])
    frontResolution =[rows, cols]
    frontImage = cv_image
    frontImage.resize([frontResolution[0], frontResolution[1], 3])
    frontImage=np.rot90(frontImage, 2)
    frontImage=np.fliplr(frontImage) 
    frontImage=cv2.cvtColor(frontImage, cv2.COLOR_RGB2BGR) 
#frongimg = cv2.cvtColor(self.frongimg, cv2.COLOR_RGB2BGR)   
    cv2.imshow('left Camera Guys...', frontImage)
    timer = cv2.waitKey(5) & 0xFF
    if timer == 27:
       pass

def get_image():
    rospy.Subscriber("/stereo/left/image_raw", Image, image_callback, queue_size=1)
    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()



if __name__=="__main__":
   window=Tk()
   window.resizable(0,0)
   window.geometry("320x320")
   window.title("DVRK CONTROL")
   window.configure(background="gray")
   pub_closebtnPsm2 =  Button(window, text="published close PSM2",command=publishDesiredjoint)
   pub_closebtnPsm2.grid(column=1, row=1)
   sub_closebtnpsm2 =  Button(window, text="subscribe close PSM2",command=sub_desired_pose)
   sub_closebtnpsm2.grid(column=2, row=1)
   pub_closebtnPsm1 =  Button(window, text="published close PSM1",command=publishDesiredjointPSM1)
   pub_closebtnPsm1.grid(column=1, row=2)
   sub_closebtnpsm1 =  Button(window, text="subscribe close PSM1",command=sub_desired_pose_PSM1)
   sub_closebtnpsm1.grid(column=2, row=2)
   #open mushua Hand...
   pub_openbtnPsm2 =  Button(window, text="published open PSM2",command=publishjointdesiredopen)
   pub_openbtnPsm2.grid(column=1, row=3)
   sub_openbtnpsm2 =  Button(window, text="subscribe open PSM2",command=sub_desired_pose)
   sub_openbtnpsm2.grid(column=2, row=3)
   pub_openbtnPsm1 =  Button(window, text="published open PSM1",command=publishjointdesiredopen_PSM1)
   pub_openbtnPsm1.grid(column=1, row=4)
   sub_openbtnpsm1 =  Button(window, text="subscribe open PSM1",command=sub_desired_pose_PSM1)
   sub_openbtnpsm1.grid(column=2, row=4)

   window.mainloop()
   
   #sub_current_pose()
   #sub_currentPSM1_pose()
   #sub_desired_pose()
   #publishDesiredjoint()
   #publishjointdesiredclose()
   #set_desired_position_callback()
   #publishDesiredjoint()
   #sub_current_pose()
   #get_image() 
   #computeM()
   #ScrewMatrix()
   #computeThetas()
   #sub_desired_pose_PSM1()
   #publishDesiredjointPSM1()
   #publishjointdesiredclose_PSM1()
   #computeThetas_PSM1()
   #sub_current_pose_PSM1()
   #ScrewMatrix_PSM1()
      
