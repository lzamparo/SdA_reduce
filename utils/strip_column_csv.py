#! /usr/bin/env python

# Take all files in sys.argv[1] ending in sys.argv[2].  They will be CSV files. Remove the data-column specified in sys.argv[3] (index from 0)
# Write the output to the same filename but with a .label extension


import sys, os

files = os.listdir(sys.argv[1])
suffix = sys.argv[2]
column_index_to_del = sys.argv[3]

os.chdir(sys.argv[1])
for f in files:
	f = f.strip()	
	
	if not f.endswith(suffix):
		continue
	
	filebits = f.split(".")
	outfile = filebits[0] + "." + "label"
	input_file = open(f,"r")
	output = open(outfile,"w")

	# read the input, truncate, write to output
	
	for line in input_file:
		line = line.strip()
		old_data = line.split(",")
		new_data = []
		front = old_data[0:int(column_index_to_del)]
		back = old_data[int(column_index_to_del)+1:len(old_data)]
		if (len(front) > 0):
			new_data.extend(front)
			new_data.extend(back)
		else:
			new_data.extend(back)		

		print >> output, ",".join(["%s" % el for el in new_data])

	# close both files
	input_file.close()
	output.close()

