class Rectangle:
    def __init__(self, width, height):
        self.width = width  # 矩形的宽度
        self.height = height  # 矩形的高度

    def area(self):
        """计算矩形的面积"""
        return self.width * self.height

    def perimeter(self):
        """计算矩形的周长"""
        return 2 * (self.width + self.height)

def main():
    # 创建一个矩形对象
    rect = Rectangle(5, 3)

    # 输出矩形的面积和周长
    print(f"矩形的面积: {rect.area()}")
    print(f"矩形的周长: {rect.perimeter()}")

if __name__ == "__main__":
    main()