import json

filehandle = open("test.txt", "r")

for line in filehandle:
    print(line.strip())

print("Break".center(20, "!"))

content = filehandle.read()

# at this point the file is already read and content is 0
print(len(content))

jsonfile = open("test.json", "r")

parsed_json = json.load(jsonfile)

print(parsed_json["somekey"])