import re
x = []
y = []
z = []
with open("Q_dist.out") as f:
	for line in f:
		data = re.search('([-+]?\d+[\.]?\d*) ([-+]?\d+[\.]?\d*)',line)
		x.append(float(data.group(1)))
		y.append(float(data.group(2)))
def cumulative(lmt,dt,y):
	cu = 0
	for i in range(0,lmt):
		cu += y[i]*dt
	return cu
dt = x[1]-x[0]
for i in range(0,len(x)):
	z.append(cumulative(i,dt,y))
	
f = open("Q_cumulative.out",'w')
for i in range(0,len(x)):
	f.write(str(x[i])+" "+str(z[i])+"\n")
f.close()
