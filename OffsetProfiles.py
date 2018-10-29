import numpy as np
import math
import cv2.ximgproc 
from matplotlib import pylab as plt
import scipy as sp
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.signal import find_peaks_cwt
import matplotlib.cm as cm

numHolder=1


plotIt=False
GBP=False #gain bad pix correction
BLF=False #bilaterial filter and sobel operations
SOB=False
Fields=np.array(["StripOffsets/Strip_offsets_IMRT_Ex_2_E_6_MU_10_Sub_LE_S1_1","StripOffsets/Strip_offsets_IMRT_Ex_1_E_6_MU_10_Sub_LE_S1_67"
,"StripOffsets/Strip_offsets_IMRT_Ex_0_E_6_MU_10_Sub_LE_S1_75","StripOffsets/Strip_3mm_IMRT_E_6_MU_10_Sub_LE_S1_76"])
Shifts=np.array([17.05,47.6,51.0,51.0])
plotIlluminated=False
LivePlot=False

PlotEdgeProfile=True

pixels_x=4096
pixels_y=4096
pixels=4096*4096

gains=np.fromfile("FlatFieldGains.raw", dtype='float64', sep="")
gains=gains.reshape([1,4096])

dark=np.fromfile("Pedestal/ped_10_27_LE.raw", dtype='<i2', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
dark=dark.reshape(1,pixels)

distanceCalibration=1/0.6072
colors = iter(cm.rainbow(np.linspace(0, 1, 6)))
# fig2,ax2=plt.subplots()
# fig3,ax3=plt.subplots()

leaf_of_choice=10

std_file=np.ones(1024*1024)
std_file=std_file*20

std_file=std_file.reshape(1024,1024)
errorOnPixIntensity=15


# leaf_gradients_dict={}
def line(x,m,c):
  return m*x+c

def gaus(x,a,x0,sigma):
  return a*exp(-(x-x0)**2/(2*sigma**2))

def lognorm(x,a,mu,sigma):
  b = (math.log(x) - mu)/math.sqrt(2*sigma**2)
  p = a*(0.5 + 0.5*math.erf(b))
  return p

def gainsBadPix(origIm, gainIm, pedIm,frames):
  # pedIm=pedIm.reshape(1,pixels)

  origIm=origIm.reshape([frames,pixels])

  #pedestal subtraction
  origIm=origIm-pedIm

  print "ped subtraction"

  origIm=origIm.reshape([frames,pixels_y,pixels_x])
  origIm[:,1022,:]=(origIm[:,1021,:]+origIm[:,1024,:])/2
  origIm[:,1023,:]=(origIm[:,1022,:]+origIm[:,1024,:])/2
  origIm[:,138,:]=(origIm[:,137,:]+origIm[:,141,:])/2
  origIm[:,139,:]=(origIm[:,138,:]+origIm[:,141,:])/2
  origIm[:,140,:]=(origIm[:,139,:]+origIm[:,141,:])/2
  origIm[:,2599,:]=(origIm[:,2598,:]+origIm[:,2602,:])/2
  origIm[:,2600,:]=(origIm[:,2599,:]+origIm[:,2602,:])/2
  origIm[:,2601,:]=(origIm[:,2600,:]+origIm[:,2602,:])/2

  print "bad cols corrected"

  #"firmware lines"
  origIm[:,4095,:]=origIm[:,4094,:]
  origIm[:,3967,:]=(origIm[:,3966,:]+origIm[:,3968,:])/2
  origIm[:,3839,:]=(origIm[:,3838,:]+origIm[:,3840,:])/2
  origIm[:,3711,:]=(origIm[:,3710,:]+origIm[:,3712,:])/2
  origIm[:,3583,:]=(origIm[:,3582,:]+origIm[:,3584,:])/2
  origIm[:,3455,:]=(origIm[:,3454,:]+origIm[:,3456,:])/2
  origIm[:,3327,:]=(origIm[:,3326,:]+origIm[:,3328,:])/2
  origIm[:,3199,:]=(origIm[:,3198,:]+origIm[:,3200,:])/2
  origIm[:,3071,:]=(origIm[:,3070,:]+origIm[:,3072,:])/2
  origIm[:,2943,:]=(origIm[:,2942,:]+origIm[:,2944,:])/2
  origIm[:,2815,:]=(origIm[:,2814,:]+origIm[:,2816,:])/2
  origIm[:,2687,:]=(origIm[:,2686,:]+origIm[:,2688,:])/2
  origIm[:,2559,:]=(origIm[:,2558,:]+origIm[:,2560,:])/2
  origIm[:,2431,:]=(origIm[:,2430,:]+origIm[:,2432,:])/2
  origIm[:,2303,:]=(origIm[:,2302,:]+origIm[:,2304,:])/2
  origIm[:,2175,:]=(origIm[:,2174,:]+origIm[:,2176,:])/2
  origIm[:,2047,:]=(origIm[:,2046,:]+origIm[:,2048,:])/2
  origIm[:,1919,:]=(origIm[:,1918,:]+origIm[:,1920,:])/2
  origIm[:,1791,:]=(origIm[:,1790,:]+origIm[:,1792,:])/2
  origIm[:,1663,:]=(origIm[:,1662,:]+origIm[:,1664,:])/2
  origIm[:,1535,:]=(origIm[:,1534,:]+origIm[:,1536,:])/2
  origIm[:,1407,:]=(origIm[:,1406,:]+origIm[:,1408,:])/2
  origIm[:,1279,:]=(origIm[:,1278,:]+origIm[:,1280,:])/2
  origIm[:,1151,:]=(origIm[:,1150,:]+origIm[:,1152,:])/2
  origIm[:,1023,:]=(origIm[:,1022,:]+origIm[:,1024,:])/2
  origIm[:,895,:]=(origIm[:,894,:]+origIm[:,896,:])/2
  origIm[:,767,:]=(origIm[:,766,:]+origIm[:,768,:])/2
  origIm[:,639,:]=(origIm[:,638,:]+origIm[:,640,:])/2
  origIm[:,511,:]=(origIm[:,510,:]+origIm[:,512,:])/2
  origIm[:,383,:]=(origIm[:,382,:]+origIm[:,384,:])/2
  origIm[:,255,:]=(origIm[:,254,:]+origIm[:,256,:])/2
  origIm[:,127,:]=(origIm[:,126,:]+origIm[:,128,:])/2

  print "firmware lines corrected"

  gainMean=gainIm.mean()
  gainIm=np.tile([gainIm],(frames,1,1))
  origIm=origIm/gains*gainMean
  origIm=origIm.reshape(frames,1024,4,1024,4).mean((2,4))

  print "gain corrected"

  return origIm

def BLFilt(im, frames):
  print im.shape
  for frame in range(0,frames):
    if (frame%20==0):
      print "frame = "+str(frame)

    startFrame=0+(1024*1024)*frame
    endFrame=0+(1024*1024)*(frame+1)

    singleFrameIm=im[startFrame:endFrame]
    singleFrameIm=singleFrameIm.reshape([1024,1024])

    singleFrameIm = np.array(singleFrameIm, dtype = np.float32)

    #bilateral filter to remove noise while preserving edges
    singleFrameIm=cv2.bilateralFilter(singleFrameIm,50,3500,2000)
    singleFrameIm=cv2.bilateralFilter(singleFrameIm,10,1000,2000)
    im[startFrame:endFrame]=singleFrameIm.ravel()
  print im.shape
  return im

def Sob_op(im, frames):
  print im.shape
  for frame in range(0,frames):
    if (frame%20==0):
      print "frame = "+str(frame)

    startFrame=0+(1024*1024)*frame
    endFrame=0+(1024*1024)*(frame+1)

    singleFrameIm=im[startFrame:endFrame]
    singleFrameIm=singleFrameIm.reshape([1024,1024])

    sobel_kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobelSigned = ndimage.convolve(singleFrameIm, sobel_kernel)
    sobelAbs=np.absolute(sobelSigned)
    sobelAbs2 = np.array(sobelAbs, dtype = np.float32)

    sobelAbs2=cv2.bilateralFilter(sobelAbs2,24,5000,5000)
    im[startFrame:endFrame]=sobelAbs2.ravel()
  print im.shape
  return im

for field in range(1,2):#0,len(Fields)
  colorChoice=next(colors)

  # if (Fields[field]==8 or Fields[field]==20 or Fields[field]==30 or Fields[field]==40):
    # continue
  print ("field ",Fields[field])
  frames=0
  if (GBP):
    im=np.fromfile(str(Fields[field])+".raw", dtype='<i2', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
    frames=len(im)/pixels

    im=gainsBadPix(im,gains,dark,frames)
    print "gain and bad pix corrected"
    print ("dtype",im.dtype)
    im.tofile(str(Fields[field])+"GBPcorr.raw")
    continue

  #iterate over frames 22 - 41 for 8MU
 
  if BLF:
    im=np.fromfile(str(Fields[field])+"GBPcorr.raw", dtype='float64', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
    frames=len(im)/1024/1024
    im=BLFilt(im, frames)
    im.tofile(str(Fields[field])+"GBPcorrBL.raw")
    continue
  else:
    im=np.fromfile(str(Fields[field])+"GBPcorrBL.raw", dtype='float64', sep="")
    print "loaded Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBL.raw"
    frames=len(im)/1024/1024

  # if SOB:
  #   im=np.fromfile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBL"+"_Sub_LE.raw", dtype='float64', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
  #   frames=len(im)/1024/1024
  #   im=Sob_op(im, frames)
  #   im.tofile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBLSOB"+"_Sub_LE.raw")
  #   continue
  # else:
  #   im=np.fromfile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBLSOB"+"_Sub_LE.raw", dtype='float64', sep="")
  #   print "loaded Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBLSOB"+"_Sub_LE.raw"
  #   frames=len(im)/1024/1024
  if LivePlot:
    plotIt=True

  nLeaves=16

  leaf_pos_dict={}
  leaf_pos_dict2={}
  leaf_pos_error_dict={}
  leaf_pos_error_dict2={}
  leaf_width_dict={}
  leaf_n_points_dict={}
  leaf_n_points_dict2={}
  for leaf in range(0,nLeaves):
    leaf_pos_dict[leaf]=np.array([])
    leaf_pos_dict2[leaf]=np.array([])
    leaf_pos_error_dict[leaf]=np.array([])
    leaf_pos_error_dict2[leaf]=np.array([])
    leaf_width_dict[leaf]=np.array([])
    leaf_n_points_dict[leaf]=np.array([])
    leaf_n_points_dict2[leaf]=np.array([])


  frame=0
  startFrame=0+(1024*1024)*frame
  endFrame=0+(1024*1024)*(frame+1)

  singleFrameIm=im[startFrame:endFrame]
  singleFrameIm=singleFrameIm.reshape([1024,1024])  


  if (plotIlluminated):
    xAxis=np.arange(1,1025,1)

    plt.plot(xAxis, singleFrameIm[...,484],'b-')
    plt.plot(xAxis, singleFrameIm[...,694],'m-')
    plt.pause(0.001)
    raw_input("end view")
    # plt.savefig("slices_Ex0_484_694.png")

  sobel_kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  sobelSigned = ndimage.convolve(singleFrameIm, sobel_kernel)
  # sobelSigned.tofile("leafPostSob.raw", sep="")


  sobelAbs=np.absolute(sobelSigned)
  # sobelAbs=cv2.bilateralFilter(sobelAbs,30,100,2000)
  # sobelAbs.tofile("leafPostSobAbs.raw", sep="")

  sobelAbs = np.array(sobelAbs, dtype = np.float32)
  sobelAbs=cv2.bilateralFilter(sobelAbs,24,5000,5000)
  # sobelAbs.tofile("leafPostSobAbsBF.raw", sep="")

  #iterate over each leaf in a frame
  start=0#36 #110 #196/4
  leafWidth=1#53
  nSlicesUsed=1#25

  # for leaf in range(0,100):
  #   for sliceNum in range(0,nSlices):
  #     #choose y slices to use for each leaf
  #     sliceStart=start+1*sliceNum
  #     sliceEnd=sliceStart+1

  #     sobelSlice=sobelAbs[...,sliceStart:sliceEnd]
  #     sobelSlice=sobelSlice.sum(axis=1,keepdims=True)

  #     sobelSliceSigned=singleFrameIm[...,sliceStart:sliceEnd]
  #     sobelSliceSigned=sobelSliceSigned.sum(axis=1,keepdims=True)

  #     #turn back to true 1D array
  #     sobelSlice=sobelSlice.ravel()
  #     sobelSliceSigned=sobelSliceSigned.ravel()

  #     xDat=np.arange(sobelSlice.size)
  # std_file=np.fromfile("stanDevs1024.raw",dtype='uint16', sep="")
 

  for leaf in range(0,1024):
    #choose y slices to use for each leaf
    sliceStart=start+leaf*leafWidth
    sliceEnd=sliceStart+nSlicesUsed
    sobelSlice=sobelAbs[...,sliceStart:sliceEnd]
    std_Slice=std_file[...,sliceStart:sliceEnd]
    sobelSlice=sobelSlice.sum(axis=1,keepdims=True)
    std_Slice=std_Slice.sum(axis=1,keepdims=True)

    if (leaf%100==0):
      print leaf    
    sobelSliceSigned=sobelSigned[...,sliceStart:sliceEnd]
    sobelSliceSigned=sobelSliceSigned.sum(axis=1,keepdims=True)

    #turn back to true 1D array
    sobelSlice=sobelSlice.ravel()
    sobelSliceSigned=sobelSliceSigned.ravel()
    std_Slice=std_Slice.ravel()
    maxsob=sobelSliceSigned.max()

    xDat=np.arange(sobelSlice.size)

    #find all peaks
    indices = find_peaks_cwt(sobelSlice, np.arange(1,20), noise_perc=30, max_distances=None, gap_thresh=None, min_length=10, min_snr=0.8)#, gap_thresh=None, min_length=10, min_snr=0.8, max_distances=None
    arraySigma=np.ones(1024)
    arraySigma=arraySigma*5
    if plotIt:
      # plt.close()

      # plt.figure(1)
      h1=plt.subplot(2, 1, 1)
      std_Slice=std_Slice*3
      # plt.plot(xDat,sobelSlice,'g-')
      plt.errorbar(xDat,sobelSlice,yerr=std_Slice, linestyle='None')
      # plt.plot(indices, sobelSlice[indices],'r^')
      # plt.pause(0.01)
      # raw_input("enter")
      # t1=np.arange(168,190,0.2) #def chisquare(x,width,height,shift):
      # arraylog=np.array([])
      # for num in t1:
      #   arraylog=np.append(arraylog,chisquare(num,4,6.3,30,174))
      # plt.plot(t1,arraylog,'r-')
      axes = plt.gca()
      # axes.set_xlim([90,230] )      
      axes.set_xlim([400,600] )
      # axes.set_ylim([-10,5000] )
      plt.rc('grid', linestyle="-", color='black')
      raw_input("hold")
      plt.pause(0.1)



    if indices.size==0:
      # print "continuing"
      continue
    #Use results of peak finder to fit gaussians and place constraint on gaussians found
    for ind in range(0,indices.size):
      fitSuccess=True
      if (sobelSlice[indices[ind]]<maxsob/2.0): #2000 for strips
        continue
      mean=indices[ind]
      # print ("mean",mean)
      pos_or_neg=sobelSliceSigned[int(np.floor(mean))]

      rangePoN=5
      if (mean>(1024-rangePoN)or mean<rangePoN):
        pos_or_neg=0
      else:
        for pon in range(0,5):
          pos_or_neg+=sobelSliceSigned[int(np.floor(mean))+pon]+sobelSliceSigned[int(np.floor(mean))-pon]

      if (pos_or_neg>0): #left leaf
        lowerWindow=8
        upperWindow=7
      else:
        lowerWindow=7
        upperWindow=8

      sigmaParam=10

      #create window to fit peak
      if (mean-lowerWindow<0):
        lower=0
      else:
        lower=mean-lowerWindow
      if (mean+upperWindow>1024):
        upper=1024
      else:
        upper=mean+upperWindow
      # if (mean-2*sigmaParam<0):
      #   lower=0
      # else:
      #   lower=mean-2*sigmaParam
      # if (mean+2*sigmaParam>1024):
      #   upper=0
      # else:
      #   upper=mean+2*sigmaParam
      # print mean
      # print (lower,upper)

      xDatSub=xDat[lower:upper]

      sobelSliceSub=sobelSlice[lower:upper]
      stdSliceSub=std_Slice[lower:upper]
      # arraySigma=np.ones(1024)
      # arraySigma=arraySigma*5
      # lowerBoundsParam=np.array([2,10,10])
      # upperBoundsParam=np.array([15,80,100])
      # totalBoundsParam=np.array([lowerBoundsParam,upperBoundsParam])

      # popt,pcov = curve_fit(gaus,xDatSub,sobelSliceSub,p0=[5.5,mean,sigmaParam],sigma=stdSliceSub)

      try :
        popt,pcov = curve_fit(gaus,xDatSub,sobelSliceSub,p0=[0.9*maxsob,mean,sigmaParam],sigma=stdSliceSub)
        # popt,pcov = curve_fit(chisquare,xDatSub,sobelSliceSub,p0=[6.0,6.0,30,mean], bounds=[[3.0,6.0,5,10],[40,70.0,100,1000]]) # x,k,width,height,shift
      except:
        popt=[0,0,0]
        # popt=[3,1,1,1000]
        pcov=0
        fitSuccess=False



      # if (mean<180 and mean>170):
      #   print "here"
      #   popt,pcov = curve_fit(chisquare,xDatSub,sobelSliceSub,p0=[4.0,6.0,30,mean], bounds=[[3.0,2.0,5,10],[40,70.0,100,1000]]) # x,k,width,height,shift
      # print popt

      #constrains on good gaussian fits
      # print ("width",popt[2])
      # print ("height",popt[0])
      # raw_input("continue")
      # print (fitSuccess, popt[1], popt[0], popt[2])

      if(abs(popt[2])<40 and abs(popt[2])>4 and abs(popt[0])>750): #popt[0]>2000 for strips
       # plt.figure(1)

        # print "conditions satisfied"

        # print popt[1]

        # print pos_or_neg
        if(plotIt):
        #   plt.plot(xDat,sobelSlice,'g-')
          h1=plt.subplot(2, 1, 1)

          plt.plot(indices, sobelSlice[indices],'r^')
          # plt.figure(1)
          if (pos_or_neg>0):
            h1=plt.subplot(2, 1, 1)

            plt.plot(xDatSub,gaus(xDatSub,*popt),'go:',label='fit')
            plt.plot(popt[1],popt[0],'b^',markersize=10)

          else:
            h1=plt.subplot(2, 1, 1)

            plt.plot(xDatSub,gaus(xDatSub,*popt),'ko:',label='fit')
            plt.plot(popt[1],popt[0],'m^',markersize=10)

            # axes = plt.gca()
            # axes.set_xlim([50,300])
            # plt.rc('grid', linestyle="-", color='black')
            # plt.savefig("FitsGaus/FitsLeafShape"+str(leaf)+".png")
            plt.pause(0.1)
            # # raw_input("continue")
            # plt.close()
          # raw_input(leaf)

        #look at peak+/-1 to see if pos or neg in signed sobel image

        # raw_input("enter")

        if (pos_or_neg<0 and fitSuccess):
          # if frame==5:
          # print leaf_pos_dict[leaf]
          leaf_pos_dict[numHolder]=np.append(leaf_pos_dict[numHolder],(popt[1])*0.0145*4*distanceCalibration-Shifts[field]) #+16.65+31.85 *0.0145 for mm-16.95
          # leaf_width_dict[numHolder]=np.append(leaf_width_dict[numHolder],popt[0]*0.0145*4*distanceCalibration)
          leaf_pos_error_dict[numHolder]=np.append(leaf_pos_error_dict[numHolder],pcov[1,1]*0.0145*4*distanceCalibration) #pcov[2,2]*0.0145*4/distancestoSensor* 0.0145*4*distanceCalibration
          leaf_n_points_dict[numHolder]=np.append(leaf_n_points_dict[numHolder],sliceStart) #s *0.023 for econds

        elif (pos_or_neg>0 and fitSuccess):
          # if frame==5:
          # print leaf_pos_dict[leaf]
          leaf_pos_dict2[numHolder]=np.append(leaf_pos_dict2[numHolder],popt[1]*0.0145*4*distanceCalibration-Shifts[field]) #+16.65+31.85 *0.0145 for mm-16.95-(popt[1]*0.0145*4/distancestoSensor)+51)
          leaf_pos_error_dict2[numHolder]=np.append(leaf_pos_error_dict2[numHolder],pcov[1,1]*0.0145*4*distanceCalibration) #pcov[2,2]*0.0145*4/distancestoSensor
          leaf_n_points_dict2[numHolder]=np.append(leaf_n_points_dict2[numHolder],sliceStart) #s *0.023 for econds     
          # print ("frame:",frame,"leaf:",leaf,"error on mean",pcov[1,1])
  # plt.plot(leaf_n_points_dict[numHolder],leaf_pos_dict[numHolder])

          # plt.figure(2)
        if LivePlot:   
          plt.subplot(2, 1, 2)
        # fig=plt.figure()
        # fig.patch.set_alpha(0.0)
          axes = plt.gca()
          # axes.set_xlim([460,500])
          # axes.set_ylim([-2.3,0.7])
          plt.errorbar(leaf_n_points_dict[numHolder],leaf_pos_dict[numHolder],fmt='m.',yerr=leaf_pos_error_dict[numHolder])
          # plt.errorbar(leaf_n_points_dict2[numHolder],leaf_pos_dict2[numHolder],fmt='b.',yerr=leaf_pos_error_dict2[numHolder])


          plt.rc('grid', linestyle="-", color='black')
          plt.grid(True)
          plt.pause(0.0001)
          # raw_input(leaf)

    if (LivePlot):
      h1.cla()

  if (PlotEdgeProfile and not LivePlot):
    fig=plt.subplots()
    # fig1=plt.subplot()

    ax = plt.gca()
    ax.set_xlim([0,1024])
    # axes.set_ylim([-2.3,0.7])
    plt.errorbar(leaf_n_points_dict[numHolder],leaf_pos_dict[numHolder],fmt='m.',yerr=leaf_pos_error_dict[numHolder])
    # plt.errorbar(leaf_n_points_dict2[numHolder],leaf_pos_dict2[numHolder],fmt='b.',yerr=leaf_pos_error_dict2[numHolder])
    # plt.plot(rangeLeaves,upperLeaves,color='c', linestyle='--', linewidth=2)
    # plt.plot(rangeLeaves,lowerLeaves,color='c', linestyle='--', linewidth=2)

    plt.rc('grid', linestyle="-", color='black')
    plt.grid(True)
    plt.pause(0.0001)
    fig.savefig(str(Fields[field])+"Edges.png")
    # fig1.cla()
    print "saved"


#     cept=0
#     try :
#       popt2,pcov2 = curve_fit(line_fixed_grad,leaf_n_points_dict[leaf],leaf_pos_dict[leaf],sigmaParam=leaf_pos_error_dict[leaf],p0=[cept])
#     except:
#       popt2=[0]
#     if plotLeaf:
#       plt.figure(2)
#       plt.plot(leaf_n_points_dict[leaf],line_fixed_grad(leaf_n_points_dict[leaf],*popt2),'r^:',label='fit')
#       plt.pause(0.001)
#       print ("cept",popt2[0],"error",pcov2[0,0])

#     frames_dict[field]=np.append(frames_dict[field],frame)
#     mean_pos_dict[field]=np.append(mean_pos_dict[field],popt2[0])
#     mean_pos_error_dict[field]=np.append(mean_pos_error_dict[field],pcov2[0,0])
# plt.figure(3)
# plt.errorbar(frames_dict[field],mean_pos_dict[field],fmt='b.',yerr=mean_pos_error_dict[field])
# plt.pause(0.001)

raw_input("Press enter to continue")


