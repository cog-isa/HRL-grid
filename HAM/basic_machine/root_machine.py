from HAM.basic_machine import basic_machine


def start(info):
    while not info["done"]:
        basic_machine.start(info)
    return info
