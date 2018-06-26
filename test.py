import itertools

def test(x=0): 
    while True: 
        x += 1
        yield x


def main():
    a = test(1)

    for _ in itertools.repeat(None, 10): 
        print(next(a))

if __name__ == '__main__':
    main()