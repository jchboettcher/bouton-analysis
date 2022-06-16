import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from scipy.spatial.distance import cdist
from scipy.stats import sem
from paths import getPath
import matplotlib.image as mpimg

def getBoutons(ind,boutonPercentile=25,show=False,save=False):
  path = getPath(*[int(i) for i in ind.split("-")])
  img_red = cv2.imread(path+"colorimgs/AVG_file000_chan0.png",0).astype('float')
  img_green = cv2.imread(path+"colorimgs/AVG_file000_chan1.png",0).astype('float')
  cellmask = np.loadtxt(path+"before/FinalMask.csv",delimiter=',').astype('int')
  manual = np.loadtxt(path+"before/manual.csv",delimiter=',').astype('int')
  if "RNTS" in path:
    cellmask = np.flip(cellmask,0)
  params = np.loadtxt(path+"params.csv",delimiter=",").astype('float')
  red_minus1,min1,max1,size1,red_minus2,min2,max2,size2 = params
  img = img_green - np.minimum(img_red*red_minus1,img_green)
  img2 = img_green - np.minimum(img_red*red_minus2,img_green)
  edges = cv2.Canny(np.uint8(img),min1,max1)
  edges2 = cv2.Canny(np.uint8(img2),min2,max2)

  totnumPixs = []

  labels = measure.label(edges, connectivity=2, background=0)
  for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
      continue
    # otherwise, construct the label mask and count the
    # number of pixels 
    labelMask = np.zeros(edges.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels < size1:
      totnumPixs.append(numPixels)
  first = len(totnumPixs)
  print(first)

  labels = measure.label(edges2, connectivity=2, background=0)
  for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
      continue
    # otherwise, construct the label mask and count the
    # number of pixels 
    labelMask = np.zeros(edges2.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels < size2:
      totnumPixs.append(numPixels)
  print(len(totnumPixs)-first)

  totnumPixs.sort()
  percThresh = totnumPixs[int(len(totnumPixs)*boutonPercentile/100)]

  totnumPixs2 = []

  labels = measure.label(edges, connectivity=2, background=0)
  mask = np.zeros(edges.shape, dtype="uint8")
  dots = np.zeros(edges.shape, dtype="uint8")
  dotsout = np.zeros(edges.shape, dtype="uint")
  dotsExpanded = np.zeros(edges.shape, dtype="uint8")
  counter = 1
  counterout = 1
  for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
      continue
    # otherwise, construct the label mask and count the
    # number of pixels 
    labelMask = np.zeros(edges.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels < size1 and numPixels >= percThresh:
      totnumPixs2.append(numPixels)
      mask = cv2.add(mask, labelMask)
      pts = np.nonzero(labelMask)
      center = np.round(np.mean(pts,axis=1))
      dots[int(center[0])][int(center[1])] = counter
      addtoout = True
      for i in range(-2,3):
        for j in range(-2,3):
          if center[0]+i < 0 or center[0]+i >= labelMask.shape[0] or center[1]+j < 0 or center[1]+j >= labelMask.shape[1]:
            continue
          if dotsout[int(center[0]+i)][int(center[1]+j)] != 0:
            addtoout = False
      if addtoout:
        dotsout[int(center[0])][int(center[1])] = counterout
        counterout += 1
      counter += 1
      for i in range(-1,2):
        for j in range(-1,2):
          if center[0]+i < 0 or center[0]+i >= labelMask.shape[0] or center[1]+j < 0 or center[1]+j >= labelMask.shape[1]:
            continue
          dotsExpanded[int(center[0])+i][int(center[1])+j] = 1
  final = np.ma.masked_where(dotsExpanded < 0.5, dotsExpanded*50)

  labels = measure.label(edges2, connectivity=2, background=0)
  mask2 = np.zeros(edges2.shape, dtype="uint8")
  dots2 = np.zeros(edges2.shape, dtype="uint8")
  dotsExpanded2 = np.zeros(edges2.shape, dtype="uint8")
  # loop over the unique components
  counter = 1
  for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
      continue
    # otherwise, construct the label mask and count the
    # number of pixels 
    labelMask = np.zeros(edges2.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels < size2 and numPixels >= percThresh:
      totnumPixs2.append(numPixels)
      mask2 = cv2.add(mask2, labelMask)
      pts = np.nonzero(labelMask)
      center = np.round(np.mean(pts,axis=1))
      dots2[int(center[0])][int(center[1])] = counter
      addtoout = True
      for i in range(-2,3):
        for j in range(-2,3):
          if center[0]+i < 0 or center[0]+i >= labelMask.shape[0] or center[1]+j < 0 or center[1]+j >= labelMask.shape[1]:
            continue
          if dotsout[int(center[0]+i)][int(center[1]+j)] != 0:
            addtoout = False
      if addtoout:
        dotsout[int(center[0])][int(center[1])] = counterout
        counterout += 1
      counter += 1
      for i in range(-1,2):
        for j in range(-1,2):
          if center[0]+i < 0 or center[0]+i >= labelMask.shape[0] or center[1]+j < 0 or center[1]+j >= labelMask.shape[1]:
            continue
          dotsExpanded2[int(center[0])+i][int(center[1])+j] = 1
  final2 = np.ma.masked_where(dotsExpanded2 < 0.5, dotsExpanded2*50)

  # _,ax = plt.subplots(1,2)
  # ax[0].imshow(img_green,cmap='gray')
  # # ax[0].imshow(mask,cmap='gray',alpha=0.3)
  # # ax[0].imshow(np.minimum(100,dots*100),cmap="autumn_r")
  # ax[0].set_xticks([])
  # ax[0].set_yticks([])
  # # ax[1].imshow(mask2,cmap='gray',alpha=0.3)
  # ax[1].imshow(img_green,cmap='gray')
  # # print(np.count_nonzero(doots2))
  # # print(np.count_nonzero(dots2))
  # y,x = np.nonzero(dotsout)
  # ax[1].plot(x,y,'y.',markersize=0.5)
  # ax[1]
  # ax[1].set_xticks([])
  # ax[1].set_yticks([])
  # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
  # plt.show()

  totnumPixs2.sort()
  totnumPixs = np.sqrt(np.array(totnumPixs)/np.pi)

  if show:
    _,ax = plt.subplots(1,1)
    ax.hist(totnumPixs,20)
    ymin, ymax = ax.get_ylim()
    ax.vlines(np.sqrt(totnumPixs2[0]/np.pi),ymin,ymax,"k",label="threshold")
    ax.set_ylim(ymin,ymax)
    plt.title("Histogram of bouton radii for "+ind)
    plt.legend()
    if save:
      plt.savefig("results/histograms/"+path[:-1]+".png")
      plt.close()
    else:
      plt.show()

    _,ax = plt.subplots(1,2)
    ax[0].imshow(img_green,cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # ax[0].set_title("Cell map")
    ax[1].imshow(img_green,cmap='gray')
    ax[1].imshow(final,cmap = 'autumn_r', interpolation='none')
    ax[1].imshow(final2,cmap = 'autumn_r', interpolation='none')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    # ax[1].set_title("Bouton map")
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    if save:
      plt.savefig("results/checkBoutons/"+path[:-1]+".png")
      plt.close()
    else:
      plt.show()

    _,ax = plt.subplots(1,2)
    ax[0].imshow(img_red,cmap='gray')
    ax[0].imshow(np.minimum(100,cellmask*100),cmap="gray",alpha=0.3)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Cell map")
    ax[1].imshow(img_green,cmap='gray')
    ax[1].imshow(final,cmap = 'autumn_r', interpolation='none')
    ax[1].imshow(final2,cmap = 'autumn_r', interpolation='none')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Bouton map")
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    if save:
      plt.savefig("results/cellsAndBoutons/"+path[:-1]+".png")
      plt.close()
    else:
      plt.show()

    # _,ax = plt.subplots(1,1)
    # # ax[0].imshow(img_red,cmap='gray')
    # # ax[0].imshow(np.minimum(100,cellmask*100),cmap="gray",alpha=0.3)
    # # ax[0].set_xticks([])
    # # ax[0].set_yticks([])
    # # ax[0].set_title("Cell map")
    # ax.imshow(img_green,cmap='gray')
    # ax.imshow(final,cmap = 'autumn_r', interpolation='none')
    # ax.imshow(final2,cmap = 'autumn_r', interpolation='none')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.set_title("Bouton map")
    # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    # plt.show()
  print(manual.shape)
  mandots = np.zeros_like(dotsout)
  for i in range(dotsout.shape[0]):
    for j in range(dotsout.shape[1]):
      mandots[i][j] = dotsout[i][j]
  for k in range(manual.shape[0]):
    y,x = np.nonzero(mandots)
    lst = []
    for i in range(y.shape[0]):
      lst.append((np.sqrt((manual[k][0]-x[i])**2+(manual[k][1]-y[i])**2),y[i],x[i]))
    lst.sort()
    for i in range(manual[k][2]):
      mandots[lst[i][1]][lst[i][2]] = 0

  y,x = np.nonzero(dotsout)
  many,manx = np.nonzero(mandots)
  # print(len(y))

  # image = mpimg.imread(path+"colorimgs/AVG_file000_chan0.png")
  # _,ax = plt.subplots(1,2,figsize=(14,7))
  # ax[0].set_xticks([])
  # ax[0].set_yticks([])
  # ax[1].set_xticks([])
  # ax[1].set_yticks([])
  # ax[0].imshow(image)
  # # ax[0].plot(x,y,'y.',markersize=1.3)
  # ax[1].imshow(image)
  # ax[1].plot(x,y,'.',color="lightblue",markersize=1.3)

  # _,ax = plt.subplots(2,2)
  # for i in range(2):
  #   for j in range(2):
  #     ax[i][j].set_xticks([])
  #     ax[i][j].set_yticks([])
  #     ax[i][j].imshow(img_green,cmap='gray')
  # # ax[0].imshow(mask,cmap='gray',alpha=0.3)
  # # ax[0].imshow(np.minimum(100,dots*100),cmap="autumn_r")
  # # ax[1].imshow(mask2,cmap='gray',alpha=0.3)
  # ax[1][0].imshow(img_red,cmap='gray')
  # ax[0][1].plot(x,y,'y.',markersize=1.3)
  # ax[1][1].plot(manx,many,'y.',markersize=1.3)

  # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
  # # plt.savefig("removeYellows/"+ind+".png")
  # plt.show()

  newdotsout = np.zeros_like(mandots)
  counter3 = 1
  for i in range(mandots.shape[0]):
    for j in range(mandots.shape[1]):
      if mandots[i][j] != 0:
        newdotsout[i][j] = counter3
        counter3 += 1
  n1 = np.nonzero(newdotsout)
  n2 = np.nonzero(mandots)
  for i in range(len(n1)):
    for j in range(len(n1[i])):
      assert n1[i][j] == n2[i][j]

  return newdotsout

def printStats(title,arr,save=False,out=None):
  n = arr.shape[0]
  if save:
    out.write(title+":\n")
    if n == 0:
      out.write("   no data!\n")
      return
    out.write("   num:  "+str(n)+"\n")
    out.write("   mean: "+str(np.mean(arr))+"\n")
    out.write("   sem:  "+str(np.std(arr)/np.sqrt(n))+"\n")
  else:
    print("   "+title+":")
    if n == 0:
      print("      no data!")
      return
    print("      num: ",n)
    print("      mean:",np.mean(arr))
    print("      sem: ",np.std(arr)/np.sqrt(n))

def printStats2(title,arr,save=False,out=None):
  out.write(title.split(" ")[0].lower()+","+",".join(arr.astype("str"))+"\n")

multiplier = 8
num = 64
legend = []
def graphStats(mat,title,col,zoom=False,save=False):
  global legend
  n = mat.shape[0]
  if n == 0:
    return
  mean = np.mean(mat,axis=0)
  stderr = sem(mat,axis=0)
  x = np.arange(multiplier,(num+1)*multiplier,multiplier).astype("float")
  if zoom:
    x = np.arange(multiplier,(num/8+1)*multiplier,multiplier).astype("float")
  x *= 509/512
  plt.errorbar(x,mean,stderr,color=col,ecolor="gray",capsize=2,elinewidth=0.5)
  if save:
    with open("5-combined-unweighted-all-rawdata.csv","a") as f:
      if title.count("mach") == 1:
        f.write("radius,")
        for i in x[:-1]:
          f.write(str(i)+",")
        f.write(str(x[-1])+"\n")
      f.write(title.lower()+" means,")
      for i in mean[:-1]:
        f.write(str(i)+",")
      f.write(str(mean[-1])+"\n")
      f.write(title.lower()+" stderrs,")
      for i in stderr[:-1]:
        f.write(str(i)+",")
      f.write(str(stderr[-1])+"\n")
  legend.append(title)

def getData(ind,terminals,percentile,save=False,getCentroids=False):
  path = getPath(*[int(i) for i in ind.split("-")])
  finalmask = np.loadtxt(path+'before/FinalMask.csv',delimiter=',').astype('int')
  if "RNTS" in path:
    finalmask = np.flip(finalmask,0)

  dim = finalmask.shape[0]

  numCells = np.max(finalmask)
  centroids = np.zeros((numCells,2))
  counts = np.zeros((numCells))

  numTerms = np.count_nonzero(terminals)
  totPixs = 0
  terms = np.zeros((numTerms,2))
  termWeights = np.zeros(numTerms)

  counter = 0
  for i in range(dim):
    for j in range(dim):
      cell = finalmask[i][j] - 1
      if cell >= 0:
        centroids[cell][0] += j
        centroids[cell][1] += i
        counts[cell] += 1
      term = terminals[i][j]
      totPixs += term
      if term > 0:
        terms[counter][0] = j
        terms[counter][1] = i
        termWeights[counter] = term
        counter += 1

  centroids[:,0] /= counts
  centroids[:,1] /= counts

  if getCentroids:
    return centroids

  dists = cdist(centroids,terms)
  # oppdists = cdist(terms,centroids)
  # print(dists.shape,oppdists.shape,np.count_nonzero(dists!=oppdists.T))
  radCounts = np.zeros((centroids.shape[0],num))
  radCounts2 = np.zeros((centroids.shape[0],num))

  for i in range(centroids.shape[0]):
    for j in range(num):
      for k in range(terms.shape[0]):
        if dists[i][k] < multiplier*(j+1):
          radCounts2[i][j] += termWeights[k]
      radCounts[i][j] = np.count_nonzero(dists[i] < multiplier*(j+1))

  mindists = np.min(dists,axis=1)
  # oppmindists = np.min(oppdists,axis=1)
  radCounts /= numTerms
  radCounts2 /= totPixs
  
  responses = np.loadtxt(path+'before/responseProfile.csv',delimiter=',').astype('int')

  return (responses,dists,radCounts,numTerms,radCounts2,totPixs,ind,percentile,path,save)

def displayData(responses,mindists,radCounts,numTerms,radCounts2,totPixs,ind,percentile,path,save):
  if not save:
    print("\n"+ind+":")
  out = None
  if save:
    out = open("closestNeuron/"+path[:-1]+".csv","w")
    # out.write("Total boutons: "+str(numTerms)+"\n")
  # print(path)
  # stomachDists = mindists[np.where(responses[:,0]==1)[0]]
  # duodenumDists = mindists[np.where(responses[:,1]==1)[0]]
  # jejunumDists = mindists[np.where(responses[:,2]==1)[0]]
  # larynxDists = mindists[np.where(responses[:,3]==1)[0]]

  allDists = np.min(mindists.T,axis=1)
  try:
    stomachDists = np.min(mindists[np.where(responses[:,0]==1)[0]].T,axis=1)
  except:
    stomachDists = np.ones_like(allDists)*-1
  try:
    duodenumDists = np.min(mindists[np.where(responses[:,1]==1)[0]].T,axis=1)
  except:
    duodenumDists = np.ones_like(allDists)*-1
  try:
    jejunumDists = np.min(mindists[np.where(responses[:,2]==1)[0]].T,axis=1)
  except:
    jejunumDists = np.ones_like(allDists)*-1
  try:
    larynxDists = np.min(mindists[np.where(responses[:,3]==1)[0]].T,axis=1)
  except:
    larynxDists = np.ones_like(allDists)*-1
  # print(numTerms)
  # print(allDists)
  # print(stomachDists)
  # print(duodenumDists)
  # print(jejunumDists)
  # print(larynxDists)

  stomach1 = radCounts[np.where(responses[:,0]==1)[0]]
  duodenum1 = radCounts[np.where(responses[:,1]==1)[0]]
  jejunum1 = radCounts[np.where(responses[:,2]==1)[0]]
  larynx1 = radCounts[np.where(responses[:,3]==1)[0]]

  stomach2 = radCounts2[np.where(responses[:,0]==1)[0]]
  duodenum2 = radCounts2[np.where(responses[:,1]==1)[0]]
  jejunum2 = radCounts2[np.where(responses[:,2]==1)[0]]
  larynx2 = radCounts2[np.where(responses[:,3]==1)[0]]

  printStats2("All cells",allDists,save,out)
  printStats2("Stomach",stomachDists,save,out)
  printStats2("Duodenum",duodenumDists,save,out)
  printStats2("Jejunum",jejunumDists,save,out)
  printStats2("Larynx",larynxDists,save,out)
  
  # plt.figure()
  # plt.title(ind+": Unweighted "+percentile+("%" if percentile != "all" else ""))
  # # graphStats(radCounts,"Total",'#1f77b4')
  # graphStats(stomach1,"Stomach",'green',save=path.count("all") == 1)
  # graphStats(duodenum1,"Duodenum",'red',save=path.count("all") == 1)
  # graphStats(jejunum1,"Jejunum",'orange',save=path.count("all") == 1)
  # graphStats(larynx1,"Larynx",'blue',save=path.count("all") == 1)
  # plt.legend(legend)
  # plt.xlabel("radius around neurons (microns)")
  # plt.ylabel("proportion of terminals")
  # if save:
  #   plt.savefig("updatedResults/unweighted-"+percentile+"/"+path[:-1]+".png")
  #   plt.close()
  # else:
  #   plt.show()
  # data = np.array([stomach1,duodenum1,jejunum1,larynx1])
  # print(stomach1.shape,duodenum1.shape,jejunum1.shape,larynx1.shape)
  # np.savetxt("4-combined-unweighted-all-rawdata.csv",data,delimiter=',',fmt='%i')

  # plt.figure()
  # plt.title(ind+": Weighted "+percentile+("%" if percentile != "all" else ""))
  # graphStats(radCounts2,"Total",'#1f77b4')
  # graphStats(stomach2,"Stomach",'#ff7f0e')
  # graphStats(duodenum2,"Duodenum",'#2ca02c')
  # graphStats(jejunum2,"Jejunum",'#d62728')
  # graphStats(larynx2,"Larynx",'#9467bd')
  # plt.legend(legend)
  # plt.xlabel("radius around neurons (microns)")
  # plt.ylabel("proportion of terminal pixels")
  # if save:
  #   plt.savefig("results/weighted-"+percentile+"/"+path[:-1]+".png")
  #   plt.close()
  # else:
  #   plt.show()

def Glp1rvsGpr65all(responses4,radCounts4,responses5,radCounts5):
  stomach4 = radCounts4[np.where(responses4[:,0]==1)[0]]
  stomach5 = radCounts5[np.where(responses5[:,0]==1)[0]]
  
  print(responses4.shape,radCounts4.shape,responses5.shape,radCounts5.shape,stomach4.shape,stomach5.shape)
  plt.figure()
  plt.title("Glp1r/Gpr65: Unweighted absolute 50 - average")
  graphStats(stomach4,"Glp1r",'g')
  graphStats(stomach5,"Gpr65",'b')
  plt.legend(legend)
  plt.xlabel("radius around neurons (microns)")
  plt.ylabel("number of terminals")
  # plt.show()
  plt.savefig("Glp1r-Gpr65-unweighted-absolute-50-avg.png")

  # plt.figure()
  # plt.title("Glp1r/Gpr65 zoomed in: Unweighted all")
  # graphStats(stomach4[:,:int(num/8)],"Glp1r",'#1f77b4',True)
  # graphStats(stomach5[:,:int(num/8)],"Gpr65",'#ff7f0e',True)
  # plt.legend(legend)
  # plt.xlabel("radius around neurons (microns)")
  # plt.ylabel("proportion of terminals")
  # # plt.show()
  # plt.savefig("Glp1rvsGpr65-zoom.png")


def Glp1rvsGpr65indiv(responseslst,radCountslst):
  stomachlist = []
  for i in range(len(responseslst)):
    response = responseslst[i]
    radCounts = radCountslst[i]
    # print(response[0],radCounts[0])
    stomachlist.append(radCounts[1][np.where(response[1][:,0]==1)[0]])
  cols = ["#dfff00","#e4d00a","#50c878","#228b22","#0047ab","#6495ed","#00ffff"]
  titles = ["4-1","4-2","4-3","4-4","5-1","5-2","5-3"]

  plt.figure()
  plt.title("Glp1r/Gpr65: Unweighted absolute 50 - individual")
  for i in range(len(stomachlist)):
    graphStats(stomachlist[i],titles[i],cols[i])
  plt.legend(legend)
  plt.xlabel("radius around neurons (microns)")
  plt.ylabel("number of terminals")
  # plt.show()
  plt.savefig("Glp1r-Gpr65-unweighted-absolute-50-indiv.png")

  # plt.figure()
  # plt.title("Glp1r/Gpr65 zoomed in: Unweighted all")
  # graphStats(stomach4[:,:int(num/8)],"Glp1r",'#1f77b4',True)
  # graphStats(stomach5[:,:int(num/8)],"Gpr65",'#ff7f0e',True)
  # plt.legend(legend)
  # plt.xlabel("radius around neurons (microns)")
  # plt.ylabel("proportion of terminals")
  # # plt.show()
  # plt.savefig("Glp1rvsGpr65-zoom.png")
