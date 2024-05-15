def testing_params(**kwargs):
    for key, value in kwargs.items():
        print(f"key:{key}, value:{value}")


if __name__ == "__main__":
    params = {"1": "first", "2": "second"}
    testing_params(**params)
