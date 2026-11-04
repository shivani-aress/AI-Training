#list

fruits = ["apple","banana", "cherry", "date", "elderberry"]
numbers = [1, 2, 3, 4, 5]
mix = fruits + numbers

print(numbers)
print(fruits)
print(mix)

mix.append("fig")
mix.insert(2, "grape")
mix.remove("banana")
print(mix)
print(mix[0:3])

#dictionary

my_dict = {
    "name": "Alice",
    "age": 30,
    "city": "New York"

}
print(my_dict)

#set

set1 = {1, 1, 1,2, 3, 4, 5}
print(set1)
unique_set2 = set(set1)
print(unique_set2)