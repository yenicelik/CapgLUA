import zipfile
import os
try:
    import wget
except:
	print("Please install the python module `wget`. For this, execute the following command: `pip install wget`")


rootA = "./Datasets/Preprocessed/DB-a"
rootB = "./Datasets/Preprocessed/DB-b"
rootC = "./Datasets/Preprocessed/DB-c"

linkA = "http://zju-capg.org/myo/data/dba-preprocessed-0"
linkB = "http://zju-capg.org/myo/data/dbb-preprocessed-0"
linkC = "http://zju-capg.org/myo/data/dbc-preprocessed-0"

def dlDataset(link, directory, no_of_subj):
	print("Start")
	if not os.path.exists(directory):
		os.makedirs(directory)
	print("It goes on")
	os.chdir(directory)
	for i in range(1, no_of_subj+1):
		if i < 10:
			dlLink = link + "0" + str(i) + ".zip"
		else:
			dlLink = link + str(i) + ".zip"
		print("Downloading: " + dlLink)
		wget.download(dlLink)
	for file in os.listdir("./"):
		try:
			print("Extracting..: " + file)
			zip = zipfile.ZipFile(file)
			zip.extractall(file[:-4])
			zip.close()
		except:
			pass
	os.chdir("../../../")

dlDataset(linkA, rootA, 18)
dlDataset(linkB, rootB, 20)
dlDataset(linkC, rootC, 10)

