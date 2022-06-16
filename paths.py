files = []

maindir = ""
with open("folders.txt","r") as file:
  for line in file:
    if line == "\n":
      maindir = ""
    elif maindir == "":
      maindir = line.strip("\n")
      files.append((maindir,[]))
    else:
      files[-1][1].append(line.strip("\n"))

def getPath(i,j,full=True):
  folder,subfolders = files[i-1]
  if full:
    return folder+"/"+subfolders[j-1]+"/"
  return folder+"/"
