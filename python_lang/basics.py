
names = ["Luke", "Paul"]


for name in names:
    print(f'Hallo {name}')

try:
    temp = "5 degrees"
    fahr = float(temp)
except Exception as e:
    print(f'Error parsing input: {e}')


a = 2
a1 = 3.0
a2 = "hallo"

def gimme():
    return "hallo","mallo"

print(type(a), type(a1), type(a2), type(gimme))
print(a * a1)

print(gimme())

n = 0
while True:
    if n == 3:
        break
    print(n)
    n = n + 1

for i in [5,4,3,2,1]:
    print(i)
print("ballan!")


smallest = None
print("Before:", smallest)
for itervar in [3, 41, 12, 9, 74, 15]:
    if smallest is None or itervar < smallest:
        smallest = itervar
        continue
    print("Loop:", itervar, smallest)
print("Smallest:", smallest)

def expect_float(foo: float):
    return foo

print(type(None))

print(expect_float("hallo"))

print(0 == 0.0, 0 is 0.0)
print("hallo" is "hallos")
print("hallo"[2], len("hallo"))

hero = "batman"

print(hero[0:3], hero[-3:])
# same as [-3:]
print(hero[-hero.find("man"):])

print("bat" in hero)

print(hero.upper())

print(dir(hero))

print(hero.rstrip("man"),hero.strip("bat"), hero.center(20, "!"))

print("hallooo     ".strip())

