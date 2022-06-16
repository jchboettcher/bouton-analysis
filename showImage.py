from sys import argv
from paths import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if len(argv) > 1:
  ind = argv[1]
  path = getPath(*[int(i) for i in ind.split("-")])
  print(path)
  image = mpimg.imread(path+"colorimgs/Composite.png")
  _,ax = plt.subplots(1,1,figsize=(7,7))
  ax.set_xticks([])
  ax.set_yticks([])
  ax.imshow(image)
  plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
  plt.show()