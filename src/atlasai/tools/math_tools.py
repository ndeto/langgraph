from langchain_core.tools import tool

class MathTools:
    @staticmethod
    def add_numbers(a: float, b: float) -> float:
        """Adds two numbers and returns their sum."""
        return a + b

    @staticmethod
    def subtract_numbers(a: float, b: float) -> float:
        """Subtracts the second number from the first number."""
        return a - b

    @staticmethod
    def multiply_numbers(a: float, b: float) -> float:
        """Multiplies two numbers and returns the product."""
        return a * b

    @staticmethod
    def divide_numbers(a: float, b: float) -> float:
        """Divides the first number by the second number."""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    @staticmethod
    def modulo_numbers(a: int, b: int) -> int:
        """Returns the remainder after dividing the first number by the second number."""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a % b

    @staticmethod
    def power_numbers(base: float, exponent: float) -> float:
        """Raises the first number to the power of the second number."""
        return base**exponent

    @staticmethod
    def absolute_value(value: float) -> float:
        """Returns the absolute value of a number."""
        return abs(value)

    @staticmethod
    def round_number(value: float, digits: int = 0) -> float:
        """Rounds a number to the nearest integer or a given number of decimal places."""
        return round(value, digits)

    @staticmethod
    def minimum_of_two(a: float, b: float) -> float:
        """Returns the smaller of two values."""
        return min(a, b)

    @staticmethod
    def maximum_of_two(a: float, b: float) -> float:
        """Returns the larger of two values."""
        return max(a, b)

    @staticmethod
    def compare_equal(left: str, right: str) -> bool:
        """Checks whether two values are equal."""
        return left == right

    @staticmethod
    def compare_not_equal(left: str, right: str) -> bool:
        """Checks whether two values are not equal."""
        return left != right

    @staticmethod
    def compare_greater_than(left: float, right: float) -> bool:
        """Checks whether the first number is greater than the second number."""
        return left > right

    @staticmethod
    def compare_less_than(left: float, right: float) -> bool:
        """Checks whether the first number is less than the second number."""
        return left < right

    @staticmethod
    def compare_greater_or_equal(left: float, right: float) -> bool:
        """Checks whether the first number is greater than or equal to the second number."""
        return left >= right

    @staticmethod
    def compare_less_or_equal(left: float, right: float) -> bool:
        """Checks whether the first number is less than or equal to the second number."""
        return left <= right

    @staticmethod
    def logical_and(left: bool, right: bool) -> bool:
        """Returns true only if both boolean inputs are true."""
        return left and right

    @staticmethod
    def logical_or(left: bool, right: bool) -> bool:
        """Returns true if at least one boolean input is true."""
        return left or right

    @staticmethod
    def logical_not(value: bool) -> bool:
        """Inverts a boolean value."""
        return not value

    @staticmethod
    def string_contains(text: str, substring: str) -> bool:
        """Checks whether one string contains another string."""
        return substring in text

    @staticmethod
    def string_starts_with(text: str, prefix: str) -> bool:
        """Checks whether a string starts with a given prefix."""
        return text.startswith(prefix)

    @staticmethod
    def string_ends_with(text: str, suffix: str) -> bool:
        """Checks whether a string ends with a given suffix."""
        return text.endswith(suffix)

    @staticmethod
    def count_list_items(items: list[str]) -> int:
        """Returns the number of items in a list."""
        return len(items)

    @staticmethod
    def get_list_item(items: list[str], index: int) -> str:
        """Returns the item at a given index from a list."""
        return items[index]

    @staticmethod
    def sort_numbers(items: list[float], order: str = "asc") -> list[float]:
        """Sorts a list of numbers in ascending or descending order."""
        reverse = order.lower() == "desc"
        return sorted(items, reverse=reverse)


math_tools = [
    tool(MathTools.add_numbers),
    tool(MathTools.subtract_numbers),
    tool(MathTools.multiply_numbers),
    tool(MathTools.divide_numbers),
    tool(MathTools.modulo_numbers),
    tool(MathTools.power_numbers),
    tool(MathTools.absolute_value),
    tool(MathTools.round_number),
    tool(MathTools.minimum_of_two),
    tool(MathTools.maximum_of_two),
    tool(MathTools.compare_equal),
    tool(MathTools.compare_not_equal),
    tool(MathTools.compare_greater_than),
    tool(MathTools.compare_less_than),
    tool(MathTools.compare_greater_or_equal),
    tool(MathTools.compare_less_or_equal),
    tool(MathTools.logical_and),
    tool(MathTools.logical_or),
    tool(MathTools.logical_not),
    tool(MathTools.string_contains),
    tool(MathTools.string_starts_with),
    tool(MathTools.string_ends_with),
    tool(MathTools.count_list_items),
    tool(MathTools.get_list_item),
    tool(MathTools.sort_numbers),
]
