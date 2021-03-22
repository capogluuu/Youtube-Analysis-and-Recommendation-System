import os 
a = os.listdir("images/")

f = open("uzantilar.txt", "w+")

for i in a:
	f.write(i)
	f.write("\n")
f.close()