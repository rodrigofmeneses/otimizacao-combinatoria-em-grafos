import numpy as np

def weights_generator(n, bin_capacity):
    return np.random.randint(1, 2 * bin_capacity // 3, n)

if __name__ == "__main__":
    num_instances = 10
    sizes = np.arange(10, 55, 5)
    bin_capacities = np.arange(100, 550, 50)

    for i in range(1, num_instances):
        for n in sizes:
            for cap in bin_capacities:
                with open(f'instances/{n}_{cap}_{i}.txt', 'w') as f:
                    f.write(f'{n}\n')
                    f.write(f'{cap}\n')
                    for w in weights_generator(n, cap):
                        f.write(f'{w}\n')