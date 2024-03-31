def individual_serial(object) -> dict:
    return {key: str(value) for key, value in object.items()}


def list_serial(objects) -> list:
    return [individual_serial(obj) for obj in objects]


if __name__ == '__main__':
    print('abc')
