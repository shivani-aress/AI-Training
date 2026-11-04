class Calculator:
    def add(self, a, b):
        return a + b

    def substract(self, a,b):
        return a-b

    def mul(self, a, b):
        return a*b

    def divide(self, a, b):
        return a/b

cal = Calculator()

while True:
    print("Select operation:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")
    print("5. Exit")

    choice = input("Enter choice (1/2/3/4/5): ")

    if choice == '5':
        print("Exiting the calculator.")
        break

    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    if choice == '1':
        print("Result: ", cal.add(num1, num2))
    elif choice == '2':
        print("Result: ", cal.substract(num1, num2))
    elif choice == '3':
        print("Result: ", cal.mul(num1, num2))
    elif choice == '4':
        if num2 != 0:
            print("Result: ", cal.divide(num1, num2))
        else:
            print("Error: Division by zero is not allowed.")
    else:
        print("Invalid input. Please try again.")