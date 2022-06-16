from sys import argv
from analysis import *
from convexhull import *
from paths import *

if len(argv) > 1:
  # print(area_from_points([
  #   [0,0],
  #   [3,0],
  #   [3,4],
  #   [2,4],
  #   [2,2],
  #   [1,2],
  #   [1,4],
  #   [0,4],
  #   [0,0],
  # ]))
  ind = argv[1]
  # displayData(*getData(ind,getBoutons(ind,boutonPercentile=25,show=False,save=False),save=False))
  boutons = getBoutons(ind,boutonPercentile=0,show=False,save=False)
  # np.savetxt("try.csv",boutons,delimiter=',',fmt='%i')
  print(np.count_nonzero(boutons))
  # convexhull(ind,boutons,show=True)
  points,allpoints = convexhull(ind,boutons)
  # print(points)
  # print(points.shape,allpoints.shape)
  centroids = getData(ind,boutons,"",getCentroids=True)
  showHull(ind,points,allpoints,centroids)
else:
  alls = {}
  # for i in range(1,6):
  #   alls[str(i)] = np.zeros((5,3))
  # for i in range(1,6):
  #   alls[str(i)] = []
  # for ind in ["1-1","1-2","1-3","1-4","1-5","1-6","2-1","2-2","2-3","3-1","3-2","3-3","3-4","3-5","4-1","4-2","4-3","4-4","4-5","5-1","5-2","5-3"]:
  # for ind in ["5-1","5-2","5-3"]:
  # for ind in ["1-1","2-1","3-1","4-1","5-1","5-2"]:
  # for ind in ["4-5"]:
  # #   print(ind)
  #   boutons = getBoutons(ind,boutonPercentile=0,show=False,save=False)
  #   # np.savetxt("try.csv",boutons,delimiter=',',fmt='%i')
  #   # print(np.count_nonzero(boutons))
  #   points,allpoints = convexhull(ind,boutons)
  #   area = area_from_points(ind,points)
  #   alls[ind[0]].append(area)
    # centroids = getData(ind,boutons,"",getCentroids=True)
    # data = showHull(ind,points,allpoints,centroids,save=True)
    # alls[ind[0]] += data
  # with open("areas.csv","w") as f:
  #   f.write("index,all,1,2,3,4,5,6\n")
  #   for i in range(1,6):
  #     f.write(str(i)+",")
  #     f.write(str(sum(alls[str(i)]))+",")
  #     for j in range(len(alls[str(i)])-1):
  #       f.write(str(alls[str(i)][j])+",")
  #     f.write(str(alls[str(i)][-1])+"\n")
    # for i in range(1,6):
    #   f.write(str(sum(alls[str(i)]))+"," if i < 5 else "\n")
    # for i in range(0,7)
  # for key,val in alls.items():
    # if key != "5":
    #   continue
    # print(key,val)
    # val[:,2] = val[:,0]/val[:,1]
    # # print(val)
    # labels = []
    # for i in range(5):
    #   labels.append(str(int(val[i][0]))+" / "+str(int(val[i][1])))
    # organLst = ["stomach","duodenum","jejunum","larynx","all"]
    # p = plt.bar(["all"]+organLst[:-1],val[:,2])
    # plt.bar_label(p,labels)
    # p[0].set_color("grey")
    # p[1].set_color("green")
    # p[2].set_color("red")
    # p[3].set_color("orange")
    # p[4].set_color("blue")
    # plt.ylim([0,1])
    # plt.title(key+"-all: Normalized Cell Ratios")
    # plt.savefig("updated-innervation-data/figures/"+getPath(int(key),0,False)+key+"-combined.png")
    # plt.close()
    # strs = ["All:","Stomach:","Duodenum:","Jejunum:","Larynx"]
    # with open("updated-innervation-data/rawStats/"+getPath(int(key),0,False)+key+"-combined.txt","w") as f:
    #   for i in range(5):
    #     f.write(strs[i]+"\n")
    #     f.write("   "+labels[i]+"\n")
    # boutons = getBoutons(ind,boutonPercentile=0,show=False,save=False)
    # path = getPath(*[int(i) for i in ind.split("-")])
    # np.savetxt("boutonMasks/"+path[:-1]+".csv",boutons,delimiter=",",fmt="%i")
    

  # alls = { "1": {} }
  alls = { "1": {}, "2": {}, "3": {}, "4": {}, "5": {} }
  # for ind in ["1-1","1-2","1-3","1-4","1-5","1-6"]:
  # for ind in ["2-1","2-2","2-3"]:
  # for ind in ["3-1","3-2","3-3","3-4","3-5"]:
  # for ind in ["4-1","4-2","4-3","4-4","4-5"]:
  # for ind in ["5-1","5-2","5-3"]:
  for ind in ["1-1","1-2","1-3","1-4","1-5","1-6","2-1","2-2","2-3","3-1","3-2","3-3","3-4","3-5","4-1","4-2","4-3","4-4","4-5","5-1","5-2","5-3"]:
    # for percStr,perc in [("50",50)]:
    for percStr,perc in [("all",0)]:
      data = getData(ind,getBoutons(ind,boutonPercentile=perc),percStr,save=True)
      displayData(*data)
      (responses,mindists,radCounts,numTerms,radCounts2,totPixs,ind,percentile,path,save) = data
      print(numTerms,totPixs)
      i1,i2 = ind.split("-")
      # if i2 == "1":
      #   folderPath = getPath(int(i1),int(i2),False)
      #   alls[i1][percStr] = [responses,mindists,radCounts,numTerms,radCounts2,totPixs,i1+"-all",percentile,folderPath+i1+"-all/",save]
      # else:
      #   for i in [0,1,2,4]:
      #     alls[i1][percStr][i] = np.concatenate((alls[i1][percStr][i],data[i]))
      #   for i in [3,5]:
      #     alls[i1][percStr][i] += data[i]
  # for key in alls:
  #   for args in alls[key].values():
  #     displayData(*args)
  
  # alls = { "4": {}, "5": {} }
  # responseslst,radCountslst = [],[]
  # for ind in ["4-1","4-2","4-3","4-4","5-1","5-2","5-3"]:
  # # for ind in ["4-3","5-1"]:
  #   for percStr,perc in [("50",50)]:
  #     data = getData(ind,getBoutons(ind,boutonPercentile=perc,show=(perc==50),save=True),percStr,save=True)
  #     (responses,mindists,radCounts,numTerms,radCounts2,totPixs,ind,percentile,path,save) = data
  #     responseslst.append((ind,responses))
  #     radCountslst.append((ind,radCounts))
  #     print(numTerms,totPixs)
  #     i1,i2 = ind.split("-")
  #     if i2 == "1":
  #       folderPath = getPath(int(i1),int(i2),False)
  #       alls[i1][percStr] = [responses,mindists,radCounts,numTerms,radCounts2,totPixs,i1+"-all",percentile,folderPath+i1+"-all/",save]
  #     else:
  #       for i in [0,1,2,4]:
  #         alls[i1][percStr][i] = np.concatenate((alls[i1][percStr][i],data[i]))
  #       for i in [3,5]:
  #         alls[i1][percStr][i] += data[i]
  # Glp1rvsGpr65indiv(responseslst,radCountslst)
  # (responses4,_,radCounts4,_,_,_,_,_,_,_) = alls["4"]["50"]
  # (responses5,_,radCounts5,_,_,_,_,_,_,_) = alls["5"]["50"]
  # Glp1rvsGpr65all(responses4,radCounts4,responses5,radCounts5)

