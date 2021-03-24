import json 

data = ["batman", "superman", "spiderman"]

data2 = ["wolverine", "captain america", "iron man"]

print(data)
print(type(data))
print(dir(data))

# this mutates
print(data.reverse())

print(data)

data[2] = "mr. bean"

print(data)
print(range(20))
print(range(len(data)))

# print(data[:2])

print("spiderman" in data)
# lexicographical min/max on string lists
print(max(data))
print(min(["b", "a", "x"]))


datadict = {"foo": 5, "bar": 2}

print("jha",json.dumps(datadict))

print(datadict.get("baz", "ne"))

loremtext = open("lorem.txt").read()

words = loremtext.split()
counts = dict()

for word in words:
    counts[word] = counts.get(word,0) + 1

# print(counts)

bigword = None
bigcount = None

for word,count in counts.items():
    if bigcount is None or count > bigcount:
        bigcount = count
        bigword = word

print(bigword, bigcount)


mutable_data = [1,2,3,4]

mutable_data[2] = 4

print(type(mutable_data), mutable_data)


immutable_data = (1,2,3,4)
try:
    immutable_data[2] = 4
except Exception as e:
    print(e)


print(type(immutable_data), immutable_data)
print(type(list(immutable_data)), list(immutable_data))

[x,y] = ("foo", "bar")

print(x)