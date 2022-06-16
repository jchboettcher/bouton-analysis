from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from paths import getPath
from random import random

def sortFunc(x,y,allpoints):
  return 1

# def convexhull(ind,terms,show=False):
#   allpoints = np.zeros((np.count_nonzero(terms),2))
#   counter = 0
#   for i in range(terms.shape[0]):
#     for j in range(terms.shape[1]):
#       if terms[i][j]:
#         allpoints[counter][0] = i
#         allpoints[counter][1] = j
#         counter += 1
#   numclose = 0.5
#   pts = []
#   for i in range(counter):
#     x,y = allpoints[i][0],allpoints[i][1]
#     dists = []
#     for j in range(counter):
#       if i == j:
#         continue
#       x_,y_ = allpoints[j][0],allpoints[j][1]
#       dists.append(np.sqrt((x-x_)**2+(y-y_)**2))
#     dists.sort()
#     pts.append((sum(dists[:int(numclose*(counter-1))]),x,y))
#   pts.sort()
#   # print(pts)
#   cutoffs = {
#     "1-1":0.7,
#     "1-2":0.52,
#     "1-3":0.58,
#     "1-4":0.85,
#     "1-5":0.7,
#     "1-6":0.65,
#     "2-1":0.9,
#     "2-2":0.55,
#     "2-3":0.55,
#     "3-1":0.6,
#     "3-2":0.65,
#     "3-3":0.75,
#     "3-4":0.62,
#     "3-5":0.75,
#     "4-1":0.7,
#     "4-2":0.77,
#     "4-3":0.7,
#     "4-4":0.55,
#     "4-5":0.6,
#     "5-1":0.8,
#     "5-2":0.75,
#     "5-3":0.75,
#   }
#   cutoff = cutoffs[ind]
#   points = np.array(list(map(lambda x: (x[1],x[2]),pts[:int(counter*cutoff)])))
#   hull = ConvexHull(points)
#   path = getPath(*[int(i) for i in ind.split("-")])
#   if show:
#     img_green = cv2.imread(path+"colorimgs/AVG_file000_chan1.png",0).astype('float')
#     plt.imshow(img_green,cmap='gray')
#     plt.plot(allpoints[:,1], allpoints[:,0],'y.',markersize=2)
#     for simplex in hull.simplices:
#       plt.plot(points[simplex, 1], points[simplex, 0], 'b-')
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#   return points,hull,allpoints

def getPoints(imageind):
  points = []
  with open("area-coords/Image "+str(imageind)+".txt","r") as f:
    i = 0
    for line in f:
      if i < 1:
        i += 1
        continue
      if line == "\n":
        points.append([])
        i = 0
        continue
      points[-1].append(line.strip("\n").split(" "))
  while len(points[-1]) == 0:
    points.pop(-1)
  for i in range(len(points)):
    points[i].append(points[i][0])
    points[i] = np.array(list(map(lambda x: (int(x[1])+0.0001*random(),int(x[0])+0.0001*random()),points[i])))
  return points

def convexhull(ind,terms,show=False):
  allpoints = np.zeros((np.count_nonzero(terms),2))
  counter = 0
  for i in range(terms.shape[0]):
    for j in range(terms.shape[1]):
      if terms[i][j]:
        allpoints[counter][0] = i
        allpoints[counter][1] = j
        counter += 1
  path = getPath(*[int(i) for i in ind.split("-")])
  imageind = 0
  with open("area-coords/image-inds.txt","r") as f:
    inds = []
    for line in f:
      inds.append(line.strip("\n"))
    imageind = inds.index(ind)+1
  points = getPoints(imageind)
  # for i in points:
  #   print(i)
  if show:
    img_green = cv2.imread(path+"colorimgs/AVG_file000_chan1.png",0).astype('float')
    plt.imshow(img_green,cmap='gray')
    plt.plot(allpoints[:,1], allpoints[:,0],'y.',markersize=2)
    # for simplex in hull.simplices:
    #   plt.plot(points[simplex, 1], points[simplex, 0], 'b-')
    for i in range(len(points)):
      for j in range(len(points[i])-1):
        plt.plot([points[i][j][0],points[i][j+1][0]],[points[i][j][1],points[i][j+1][1]],"b-")
    plt.xticks([])
    plt.yticks([])
    plt.show()
  return points,allpoints

def intersect(point1,point2,bouton):
  # y - y1 = (y2-y1)/(x2-x1)*(x - x1)
  # y = y3/x3*x
  # y3/x3*x = (y2-y1)/(x2-x1)*(x - x1) + y1
  # y3/x3*x = (y2-y1)/(x2-x1)*x - (y2-y1)/(x2-x1)*x1 + y1
  # (y3/x3 - (y2-y1)/(x2-x1))*x = y1 - (y2-y1)/(x2-x1)*x1
  # x = (y1 - (y2-y1)/(x2-x1)*x1)/(y3/x3 - (y2-y1)/(x2-x1))
  x1,y1 = point1
  x2,y2 = point2
  x3,y3 = bouton
  x = (y1 - (y2-y1)/(x2-x1)*x1)/(y3/x3 - (y2-y1)/(x2-x1))
  y = y3/x3*x
  # print(y,(y2-y1)/(x2-x1)*(x - x1)+y1)
  return ((y < y3 and x < x3)\
    and ((y < y1 and y > y2 and x < x1 and x > x2)\
      or (y > y1 and y < y2 and x > x1 and x < x2)\
      or (y < y1 and y > y2 and x > x1 and x < x2)\
      or (y > y1 and y < y2 and x < x1 and x > x2)))

def point_in_hull(points,bouton):
  # count = 0
  # for i in range(len(points)-1):
  #   doesit = intersect(points[i],points[i+1],bouton)
  #   if doesit[0]:
  #     # ax[1].plot([0,doesit[1][0],bouton[1]], [0,doesit[1][1],bouton[0]],'y-',linewidth=0.4)
  #     # ax[1].plot(doesit[1][0], doesit[1][1],'b.',markersize=1.7)
  #     count += 1
  # if count % 2:
  #   # print(count)
  #   return True
  # return False
  # return count % 2 == 1
  return sum([intersect(points[i],points[i+1],bouton) for i in range(len(points)-1)]) % 2 == 1

# def point_in_hull(point, hull, tolerance=1e-12):
#     return all(
#         (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
#         for eq in hull.equations)

def area_from_points(ind,points):
  tot = 0
  for j in range(len(points)):
    sum1 = sum([points[j][i][0]*points[j][i+1][1] for i in range(len(points[j])-1)])
    sum2 = sum([points[j][i+1][0]*points[j][i][1] for i in range(len(points[j])-1)])
    tot += 0.5*abs(sum1-sum2)
  return tot

def showHull(ind,points,allpoints,centroids,save=False):
  path = getPath(*[int(i) for i in ind.split("-")])
  out = None
  if save:
    out = open("updated-innervation-data/rawStats/"+path[:-1]+".txt","w")
  cellmask = np.loadtxt(path+"before/FinalMask.csv",delimiter=',').astype('int')
  if "RNTS" in path:
    cellmask = np.flip(cellmask,0)
  img_green = cv2.imread(path+"colorimgs/AVG_file000_chan1.png",0).astype('float')
  responses = np.loadtxt(path+'before/responseProfile.csv',delimiter=',').astype('int')
  organCellmasks = []
  for cellind in range(4):
    organCellmasks.append(np.zeros_like(cellmask))
    for i in range(cellmask.shape[0]):
      for j in range(cellmask.shape[1]):
        cell = cellmask[i][j]
        if cell:
          if responses[cell-1][cellind]:
            organCellmasks[cellind][i][j] = 100
  organLst = ["stomach","duodenum","jejunum","larynx","all"]
  # colors = ["autumn","winter","cool","Wistia"]
  colors = [
    ListedColormap(["green"]),
    ListedColormap(["red"]),
    ListedColormap(["orange"]),
    ListedColormap(["blue"]),
  ]
  # red, blue, light blue, yellow
  # green, red, orange, blue
  stomach = centroids[np.where(responses[:,0]==1)[0]]
  duodenum = centroids[np.where(responses[:,1]==1)[0]]
  jejunum = centroids[np.where(responses[:,2]==1)[0]]
  larynx = centroids[np.where(responses[:,3]==1)[0]]
  cellmaps = {
    "all": centroids, "stomach": stomach, "duodenum": duodenum,
    "jejunum": jejunum, "larynx": larynx,
  }
  data = np.zeros((5,3))
  fig,ax = plt.subplots(1,2,figsize=(12,6))
  ax[0].set_aspect(5.3)
  ax[1].imshow(img_green,cmap='gray')
  ax[1].set_xticks([])
  ax[1].set_yticks([])
  # for simplex in hull.simplices:
  #   ax[1].plot(points[simplex, 1], points[simplex, 0], '-',color="white",linewidth=1.2)
  for i in range(len(points)):
    for j in range(len(points[i])-1):
      ax[1].plot([points[i][j][1],points[i][j+1][1]],[points[i][j][0],points[i][j+1][0]], '-',color="white",linewidth=1.2)
  # for cells in ["all"]:
  labels = []
  for cells in ["all","no","stomach","duodenum","jejunum","larynx"]:
    capsCells = cells[0].upper()+cells[1:]
    # if cells == "no":
    if cells != "no":
      if cells == "all":
        ax[1].imshow(np.minimum(100,cellmask*100),cmap="gray",alpha=0.2)
        ax[1].plot(allpoints[:,1], allpoints[:,0],'w.',markersize=0.9)
      cellmap = cellmaps[cells]
      count = 0
      for p in cellmap:
        p = np.array([p[1],p[0]])
        # if point_in_hull(p,hull):
        #   count += 1
        if any([point_in_hull(pts,p) for pts in points]):
          count += 1
          # ax[1].plot(p[1], p[0],'y.',markersize=2.7)
      cellind = organLst.index(cells)
      data[(cellind+1) % 5][0] = count
      data[(cellind+1) % 5][1] = cellmap.shape[0]
      if cellmap.shape[0]:
        data[(cellind+1) % 5][2] = count / cellmap.shape[0]
      labels.append(str(count)+" / "+str(cellmap.shape[0]))
      if save:
        out.write(capsCells+":\n")
        out.write("   "+str(count)+" / "+str(cellmap.shape[0])+"\n")
      else:
        print(capsCells+":")
        print("  ",count,"/",cellmap.shape[0])
      if cells != "all":
        cellind = organLst.index(cells)
        mask = organCellmasks[cellind]
        # ax[1].imshow(np.ma.masked_where(mask==0,mask),cmap="binary")
      else:
        for organ in organLst:
          if organ == "all":
            continue
          cellind = organLst.index(organ)
          mask = organCellmasks[cellind]
          ax[1].imshow(np.ma.masked_where(mask==0,mask),cmap=colors[cellind])
    # title = "Area of Innervation (Showing "+capsCells+" Cells)"
    title = "Area of Innervation"
    if cells == "all":
      ax[1].set_title(title)
    # if save:
    #   plt.savefig("area-results/convexHulls/"+cells+"Cells/"+path[:-1]+".png")
    #   plt.close()
    # else:
    #   plt.show()
  # plt.bar(organLst[:-1],data[1:,0])
  # plt.title("Absolute Cell Counts")
  # if save:
  #   plt.savefig("area-results/innervationData/barGraphs/absolute/"+path[:-1]+".png")
  #   plt.close()
  # else:
  #   plt.show()
  # plt.bar(organLst[:-1],data[1:,0],color="b")
  # plt.bar(organLst[:-1],data[1:,1]-data[1:,0],bottom=data[1:,0],color="r")
  # plt.title("Abolute Cell Counts with Totals")
  # if save:
  #   plt.savefig("area-results/innervationData/barGraphs/withTotals/"+path[:-1]+".png")
  #   plt.close()
  # else:
  #   plt.show()
  p = ax[0].bar(["all"]+organLst[:-1],data[:,2])
  ax[0].bar_label(p,labels)
  p[0].set_color("grey")
  p[1].set_color("green")
  p[2].set_color("red")
  p[3].set_color("orange")
  p[4].set_color("blue")
  ax[0].set_ylim([0,1])
  ax[0].set_title(ind+": Normalized Cell Ratios")
  if save:
    plt.savefig("updated-innervation-data/figures/"+path[:-1]+".png")
    # plt.savefig("area-results/innervationData/barGraphs/normalized/"+path[:-1]+".png")
    plt.close()
  else:
    plt.show()
  return data