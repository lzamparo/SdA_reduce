# -*- coding: utf-8 -*-

# compare the structures of two hdf5 files by interpreting their h5ls dump txt tiles
import re
import sys

def parse_line(line,regex):
    ''' Each line looks like: /plates/9/81             Dataset {22/Inf, 916} '''
    match = regex.match(line.strip())
    if match:
        return [int(duder) for duder in match.groups()]
    else:
        return "barf"

firstfile = open(sys.argv[1],'r')
secondfile = open(sys.argv[2],'r')
regex = re.compile('\/plates\/([\d]{1,2})\/([\d]+) \s+ Dataset \{([\d]+)[\/]?[\w]*, 916\}', flags=0)


first_plates_dict = {}
second_plates_dict = {}

for i in xrange(1,15):
    first_plates_dict[i] = {}
    second_plates_dict[i] = {}
        
print "building first dict..."
for line in firstfile:
    line = line.strip()
    zug = parse_line(line,regex)
    if zug != "barf":
        plate,well,count = zug
        first_plates_dict[plate][well] = count
print "done!"
firstfile.close()

print "building second dict..."
for line in secondfile:
    line = line.strip()
    zug = parse_line(line,regex)
    if zug != "barf":
        plate,well,count = zug
        second_plates_dict[plate][well] = count
print "done!"
secondfile.close()

print "comparing both dicts..."
diff = set(first_plates_dict.keys()) - set(second_plates_dict.keys())
print "set difference between first, second: ", diff

count_errors = 0
well_errors = 0
for p in first_plates_dict.keys():
    for w in first_plates_dict[p].keys():
        try:
            first_count = first_plates_dict[p][w]
            second_count = second_plates_dict[p][w]
            diff = first_count - second_count
            #print "difference in first, second file counts for plate %d, well %d : %d" % (p,w,diff)
            if diff < 0:
                print "Count error: cannot have more cells post filtration"
                count_errors += 1
        except KeyError:
            print "Well error: second file does not have an entry for plate %d, well %w" % (p,w)
            well_errors += 1
            
if count_errors == 0 and well_errors == 0:            
    print "No errors at this resolution of testing."
else:
    print "Found %d count errors,  %d well errors.  You should check into this." % (count_errors, well_errors)